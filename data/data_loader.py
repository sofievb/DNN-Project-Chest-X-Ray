from torch.utils.data import DataLoader, random_split
from data.custom_image_dataset import ChestXRayDataset
import pandas as pd

class DataProcessing:
    def __init__(self, img_dirs, annot_file, transform=None, multi_label=False):
        self.img_dirs = img_dirs
        self.annot_file = annot_file
        self.transform = transform
        self.multi_label = multi_label

    def filter_images_by_txt(self, txt_file, batchsize, val_split=0.2):
        """
        Filters images based on the filenames listed in a .txt file.

        Args:
            txt_file (str): Path to the .txt file containing image names.
        
        Returns:
            pd.DataFrame: Filtered DataFrame with only the images listed in txt_file.
        """
        with open(txt_file, 'r') as f:
            image_names = set(line.strip() for line in f)

        full_df = pd.read_csv(self.annot_file)
        filtered_df = full_df[full_df['Image Index'].isin(image_names)]
        return filtered_df

    def load_train_val_data(self, txt_file, batchsize, val_split=0.2, n_workers=4):
        """
        Loads training and validation datasets based on txt file.

        Args:
            txt_file (str): Path to .txt file specifying train/val images.
        
        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation data loaders.
        """
        filtered_df = self.filter_images_by_txt(txt_file, batchsize=batchsize)
        '''
        labels = filtered_df['Finding Labels']
        if self.multi_label:
            label_matrix = labels.str.get_dummies(sep='|')
        else:
            label_matrix = labels.str.split('|').str[0]
        '''
        dataset = ChestXRayDataset(
            img_labels= filtered_df,
            img_dirs=self.img_dirs,
            transform=self.transform,
            multi_label=self.multi_label
        )

        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=n_workers)
        val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=n_workers)

        return train_loader, val_loader

    def load_test_data(self, txt_file, batchsize, n_workers=4):
        """
        Loads test dataset based on txt file.

        Args:
            txt_file (str): Path to .txt file specifying test images.
        
        Returns:
            DataLoader: Test data loader.
        """
        filtered_df = self.filter_images_by_txt(txt_file)        
        test_dataset = ChestXRayDataset(
            img_labels=filtered_df,
            img_dirs=self.img_dirs,
            transform=self.transform,
            multi_label=self.multi_label
        )

        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=n_workers)
        return test_loader
