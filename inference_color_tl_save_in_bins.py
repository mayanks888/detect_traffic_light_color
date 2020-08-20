from color_detect_model import  *
from models import  *
from color_detect_model import *
import os
# from __future__ import print_function
import argparse
import cv2
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import shutil
import random
from PIL import Image
import torch
import time
from datetime import datetime
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# weight_path='color_model_new.pt'
# weight_path='color_model_train_v4.pt'
weight_path='color_model_train_google_included.pt'
model = Color_Net_CNN(output=4)
# model = Net()
model.load_state_dict(torch.load(weight_path))

model.to(0).eval()
input_folder='/home/mayank_s/datasets/color_tl_datasets/more_google_images/crop_yolov4_google_image_tl'
recgn_img_size=32
device=0
color_list=['black', 'green', 'red','yellow']
output_folder="/home/mayank_s/Desktop/color_outputs/sample"
# if os.path.exists(output_folder):
#     shutil.rmtree(output_folder)  # delete output folder
# os.makedirs(output_folder)

# normalize = torchvision.transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
    # time_start = time.time()

    for filename in filenames:
        file_path = (os.path.join(root, filename));
        # ################################33
        img0 = Image.open(file_path)
        img0 = img0.convert("RGB")
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        img = transform(img0)
        img=img.unsqueeze(0)
        # input_batch = img0.repeat(ref_batch, 1, 1, 1)
        ############################
        img = img.to(device)
        # output = model(img)
        output = model(img)
        # print("actual time taken", (time.time() - t1) * 1000)
        data = torch.argmax(output, dim=1)
        print(output)
        light_color = color_list[int(data)]
        image=cv2.imread(file_path)
        # cv2.putText(image, light_color, (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        # cv2.putText(image, light_color, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, .30, (255, 255, 255), lineType=cv2.LINE_AA)
        ############################################################################################
        # output_path_crop1 = (os.path.join(output_folder, filename))
        # 1
        # cv2.imwrite(output_path_crop1, image)

        # ch = cv2.waitKey(10)
        # output_path_crop1 = (os.path.join(self.output_path_crop, filename))
        ts = time.time()
        st = datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S_%f')
        new_file_name = str(st) + ".jpg"

        fd = light_color
        if fd == "green":
            color_path = "green"
        elif fd == "red":
            color_path = 'red'
        elif fd == "yellow":
            color_path = "yellow"
        elif fd == "black":
            color_path = "black"
        else:
            color_path = "unknown"
        output_path_crop1 = (os.path.join(output_folder, color_path, new_file_name))
        cv2.imwrite(output_path_crop1, image)
