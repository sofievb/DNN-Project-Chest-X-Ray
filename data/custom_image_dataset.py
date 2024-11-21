import pandas as pd
import os
from torchvision.io import read_image
import torch
class ChestXRayDataset():
    def __init__(self, annot_file, img_dir, transform=None, target_transform=None, multi_label=False):
        self.img_labels = pd.read_csv(annot_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.multi_label = multi_label
        #Classification of multiple diagnoses
        if multi_label:
            all_labels = set()
            for labels in self.img_labels['Finding Labels']:
                all_labels.update(labels.split('|'))
            self.label_dict = {label: index for index, label in enumerate(sorted(all_labels))}

    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, index):
        img_fname = self.img_labels.iloc[index]['Image Index']
        img_path = os.path.join(self.img_dir, img_fname)
        img = read_image(img_path)
        img_label = self.img_labels.iloc[index]['Finding Labels']
        if self.multi_label:
        #Convert to binary vector
            label_vector = torch.zeros(len(self.label_dict), dtype=torch.float)
            for label in img_label.split('|'):
                if label in self.label_dict:
                    label_vector[self.label_dict[label]] = 1.0
            img_label = label_vector
        elif self.target_transform:
            img_label = self.target_transform(img_label)
            
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(img_label)
        return img, label
