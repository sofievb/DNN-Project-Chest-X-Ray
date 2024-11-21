from torch.utils.data import Dataset
import torchvision.transforms as tf
import pandas as pd
from torchvision.io import read_image
import os
from torch.utils.data import DataLoader
from data.custom_image_dataset import ChestXRayDataset
#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CustomImageDataset(Dataset):
    def __init__(self, annot_file, img_dir, transform=None, target_transform=None):
        #super().__init__()
        self.img_labels = pd.read_csv(annot_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_folders = [os.path.join(img_dir, f"images_{str(i).zfill(3)}") for i in range(1, 13)]
    
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        # Hent bildefilnavn og søk gjennom mappene
        img_name = self.img_labels.iloc[idx, 0]
        img_path = None
        for folder in self.image_folders:
            possible_path = os.path.join(folder, img_name)
            if os.path.exists(possible_path):
                img_path = possible_path
                break
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in any folder.")
        
        # Les inn bildet og etiketten
        image = read_image(img_path).float() / 255.0  # Normaliser til [0, 1]
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
        
class DataProcessing():
    def __init__(self, data_dir,annot_file, train=False):
        self.data_dir = data_dir
        self.annot_file = annot_file
        self.transform = self.data_transform()
        if train:
            self.transform = tf.Compose([
                self.data_augmentation(),
                self.data_transform()
            ])
        else:
            # Bruk bare data_transform for valideringsdata
            self.transform = self.data_transform()


    def load_data(self,n_workers=1, batchsize=64):

        dataset = ChestXRayDataset(annot_file=self.annot_file, img_dir=self.data_dir, transform=self.transform)        
        loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=n_workers)
        return loader
    
    def data_transform(self):
        #transformations for normalization, resizing of all data
        resize = tf.Resize((224,224)) #resize: standard resnet/vgg/alexnet
        normalization = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensors = tf.ToTensor()
        return tf.Compose([resize, normalization, tensors])
    
    def data_augmentation(self):
        #transformation for training data
        transforms = tf.Compose([
            tf.RandomHorizontalFlip(),
            tf.ColorJitter(brightness=0.2,contrast=0.2),
            tf.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
            tf.RandomRotation(degrees=10)
        ])

        return transforms
    