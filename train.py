import config
import torch
import torch.optim as optim
from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
)

from torch.utils.tensorboard import SummaryWriter
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors,writer,epoch):
    loop = tqdm(train_loader, leave=True)

    losses = []
    box_losses = []
    total_loss_obj = []
    total_loss_no_obj = []
    class_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)

            l0 = loss_fn(out[0], y0, scaled_anchors[0])
            l1 = loss_fn(out[1], y1, scaled_anchors[1])
            l2 = loss_fn(out[2], y2, scaled_anchors[2])
            loss = sum(l0) + sum(l1) + sum(l2)

            losses.append(loss.item())
            box_losses.append(l0[0]+l1[0]+l2[0])
            total_loss_obj.append(l0[1]+l1[1]+l2[1])
            total_loss_no_obj.append(l0[2]+l1[2]+l2[2])
            class_loss.append(l0[3]+l1[3]+l2[3])

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    writer.add_scalar("train/loss", mean_loss, epoch)
    writer.add_scalar("train/box_loss",  sum(box_losses) / len(loop), epoch)
    writer.add_scalar("train/total_loss_obj", sum(total_loss_obj) / len(loop), epoch)
    writer.add_scalar("train/total_loss_no_obj", sum(total_loss_no_obj) / len(loop), epoch)
    writer.add_scalar("train/class_loss", sum(class_loss) / len(loop), epoch)


def eval_fn(eval_loader, model, optimizer, loss_fn, scaler, scaled_anchors,writer,epoch):
    loop = tqdm(eval_loader, leave=True)

    losses = []
    box_losses = []
    total_loss_obj = []
    total_loss_no_obj = []
    class_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            
            out = model(x)
            l0 = loss_fn(out[0], y0, scaled_anchors[0])
            l1 = loss_fn(out[1], y1, scaled_anchors[1])
            l2 = loss_fn(out[2], y2, scaled_anchors[2])
            loss = sum(l0) + sum(l1) + sum(l2)

            losses.append(loss.item())
            box_losses.append(l0[0]+l1[0]+l2[0])
            total_loss_obj.append(l0[1]+l1[1]+l2[1])
            total_loss_no_obj.append(l0[2]+l1[2]+l2[2])
            class_loss.append(l0[3]+l1[3]+l2[3])

            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

    writer.add_scalar("eval/loss", mean_loss, epoch)
    writer.add_scalar("eval/box_loss",  sum(box_losses) / len(loop), epoch)
    writer.add_scalar("eval/total_loss_obj", sum(total_loss_obj) / len(loop), epoch)
    writer.add_scalar("eval/total_loss_no_obj", sum(total_loss_no_obj) / len(loop), epoch)
    writer.add_scalar("eval/class_loss", sum(class_loss) / len(loop), epoch)

  
def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader,IMAGE_SIZE = get_loaders()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (torch.tensor(config.ANCHORS)
        * torch.tensor([IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    writer = SummaryWriter(config.LOG_DIR)

    for epoch in range(config.NUM_EPOCHS):
        print(f"Currently epoch {epoch}")
        print('training')
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors,writer,epoch)
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
        check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)
        """
        print('check training map')
        pred_boxes, true_boxes = get_evaluation_bboxes(
                train_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
        mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
        )
        #writer.add_scalar("train/mAP", mapval.item(), epoch)
        print(mapval.item())
        """
        print('evaluating')
        
        model.eval()
        eval_fn(test_loader, model, optimizer, loss_fn, scaler, scaled_anchors,writer,epoch)
        #check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

        """
        print('check eval map')
        pred_boxes, true_boxes = get_evaluation_bboxes(
                train_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
        
        mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
        )

        #writer.add_scalar("eval/mAP", mapval.item(), epoch)
        print(mapval.item())
        """
        
        model.train()
        print()


if __name__ == "__main__":
    print(config.DEVICE)
    main()
