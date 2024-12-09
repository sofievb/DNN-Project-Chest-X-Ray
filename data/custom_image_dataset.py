import pandas as pd
import os
from torchvision.io import read_image
import torch
from PIL import Image
class ChestXRayDataset():
    def __init__(self, img_labels, img_dirs, transform=None, target_transform=None, multi_label=False):
        #self.img_labels = pd.read_csv(annot_file)
        self.img_labels = img_labels
        self.img_dirs = img_dirs
        self.transform = transform
        self.target_transform = target_transform
        self.multi_label = multi_label

        self.img_labels['Finding Labels'] = self.img_labels['Finding Labels'].astype(str)


        #Classification of multiple diagnoses
        if multi_label:
            all_labels = set()
            for labels in self.img_labels['Finding Labels']:
                all_labels.update(labels.split('|'))
            self.label_dict = {label: index for index, label in enumerate(sorted(all_labels))}
        else:
            self.img_labels = self.img_labels[~self.img_labels['Finding Labels'].str.contains(r'\|')]
            unique_labels = self.img_labels['Finding Labels'].unique()
            self.label_dict = {label: index for index, label in enumerate(sorted(unique_labels))}
        print(f"Label Mapping: {self.label_dict}")  # Debug: Vis label-mappingen
        print(f"Filtrert datasettstørrelse: {len(self.img_labels)}")  # Debug: Dataset størrelse etter filtrering

        self.image_paths = {}
        for dir_path in self.img_dirs:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths[file] = os.path.join(root, file)

    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, index):
        img_fname = self.img_labels.iloc[index]['Image Index']
        img_path = self.image_paths.get(img_fname, None)
        if img_path is None:
            raise FileNotFoundError(f"Bildet {img_fname} ble ikke funnet i noen av oppgitte paths.")

        # Les bildet som tensor
        img = Image.open(img_path).convert('RGB')

        # Anvend transformasjoner
        if self.transform:
            img = self.transform(img)

        # Få label for bildet
        img_label = self.img_labels.iloc[index]['Finding Labels']

        if self.multi_label:
        #Convert to binary vector
            label_vector = torch.zeros(len(self.label_dict), dtype=torch.float)
            for label in img_label.split('|'):
                if label in self.label_dict:
                    label_vector[self.label_dict[label]] = 1.0
            img_label = label_vector
        else:
            first_label = img_label.split('|')[0]
            img_label = self.label_dict.get(first_label, -1)

        if img_label == -1:
            raise ValueError(f"Label '{img_label}' ikke funnet i label_dict. Sjekk dataset-anmerkningene.")
 
        if self.target_transform:
            label = self.target_transform(img_label)
        
        return img, img_label