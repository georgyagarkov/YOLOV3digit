import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2

DATASET_TRAIN = r'F:\DATASET3k\Train'
DATASET_VALID = r'F:\DATASET3k\Valid'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8
BATCH_SIZE = 10
NUM_CLASSES = 2
#IMAGE_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 501
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.5
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = r"E:\yolov3\checkpoint.pth.tar"
LOG_DIR = r'E:\yolov3'

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  


scale = 1.1
"""
train_transforms = A.Compose(
    [
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.PixelDropout(),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.0037437496462906254, 0.006080065429586512, 0.1661164222399876], std=[0.0056837032233803176, 0.012132838416284545, 0.24912470847115728], max_pixel_value=0.5),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0.0037437496462906254, 0.006080065429586512, 0.1661164222399876], std=[0.0056837032233803176, 0.012132838416284545, 0.24912470847115728], max_pixel_value=0.5),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
"""


PLATE_CLASSES = [
    'OUTSIDE_BREAK',
    'SCARS',
]

