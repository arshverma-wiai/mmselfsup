from email.mime import image
import mmcv
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
import pickle

from ..builder import DATASOURCES
from .base import BaseDataSource


@DATASOURCES.register_module()
class NIH(BaseDataSource):

    base_folder = "/scratchg/data/NIH-CXR"
    # split_version = "SSL-50k"
    # split_folder = os.path.join(base_folder, "splits", split_version)
    images_folder = os.path.join(base_folder, "images")
    labels_processed = os.path.join(base_folder, "labels_processed_v2.csv")
    label2image_dict = os.path.join(base_folder, "images_per_label.json")

    name_to_feature_code_mapping = {
            "Infiltration/Consolidation": 0,
            "Emphysema": 1,
            "Edema": 2,
            "Atelectasis": 3,
            "Nodule/Mass": 4,
            "Pneumothorax": 5,
            "Fibrosis": 6,
            "Cardiomegaly": 7,
            "Hernia": 8,
            "Effusion": 9,
            "Pleural_Thickening": 10,
        }


    def load_annotations(self):
        
        self.df_labels = pd.read_csv(self.labels_processed)
        self.df_labels.set_index("Image Index", inplace=True)
        self.df_labels = self.df_labels.loc[:, [*self.name_to_feature_code_mapping]]

        if not self.test_mode:
            imagepath = os.path.join(self.split_folder, "train.txt")
        else:
            imagepath = os.path.join(self.split_folder, "test.txt")

        self.imgs = self.get_imagelist(imagepath)
        self.gt_labels = self.get_labels()

        data_infos = []

        for i, (image_name, gt_label) in enumerate(zip(self.imgs, self.gt_labels)):
            # img = self.open_image(image_name)
            # info = {'img': img, 'gt_label': gt_label, 'idx': i}
            info = {'img_type': 'cxr', 'img_info': {'filename': image_name}, 'gt_label': gt_label, 'idx': i}
            data_infos.append(info)
        
        return data_infos

    def get_imagelist(self, path):
        with open(path) as file:
            image_list = file.read().split("\n")[:-1]

        return image_list

    def get_labels(self):
        labels = []

        for image_name in self.imgs:
            label_vector = self.df_labels.loc[image_name, :].to_numpy()
            labels.append(label_vector)
        
        return labels

    def open_image(self, image_name):
        image_path = os.path.join(self.images_folder, image_name)
        temp = Image.open(image_path).convert("RGB")
        image = np.asarray(temp)

        return image