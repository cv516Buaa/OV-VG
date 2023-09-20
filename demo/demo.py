from OVVG.util.inference import load_model, load_image
import cv2
import json
import numpy as np
import os
import torch
from torchvision.ops import box_convert

# Input image and text
IMAGE_PATH = "images/000000000360.jpg"
OUTPUT_DIR = 'outputs/000000000360.jpg'
TEXT = 'a flag on the left side of the picture'
# '000000000360.jpg' 'a flag on the left side of the picture'
# '000000005193.jpg' 'an air conditioner on the wall next to the yellow surfboard'
# '000000008351.jpg' 'a white and purple dog collar on neck of right dog'
model = load_model("OVVG/config/OVVG.py", "checkpoint/0.4155.pth")

# Predict box
image_source, image = load_image(IMAGE_PATH)
pred_boxes, logits = model(image[None], captions=[TEXT])
h, w, _ = image_source.shape
pred_boxes = pred_boxes * torch.Tensor([w, h, w, h])
pred_boxes = box_convert(boxes=pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")

# Visualization Results
im = cv2.imdecode(np.fromfile(IMAGE_PATH, dtype=np.uint8), -1)
cv2.rectangle(im, (int(pred_boxes[0]), int(pred_boxes[1])), (int(pred_boxes[2]), int(pred_boxes[3])), (255, 0, 0), 2)
cv2.imwrite(OUTPUT_DIR, im)