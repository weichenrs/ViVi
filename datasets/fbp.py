import os
from PIL import Image
import numpy as np
from torchvision.datasets import VisionDataset
from utils import create_palette

class FBP(VisionDataset):
    def __init__(self,
                 root,
                 img_folder,
                 mask_folder,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root, transform=transform, target_transform=target_transform)
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.images = list(
            sorted(os.listdir(os.path.join(self.root, img_folder))))
        self.masks = list(
            sorted(os.listdir(os.path.join(self.root, mask_folder))))
        self.color_to_class = create_palette(
            os.path.join(self.root, 'class_dict.csv'))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_folder, self.images[index])
        mask_path = os.path.join(self.root, self.mask_folder,
                                 self.masks[index])

        img = Image.open(img_path).convert('RGB')
        labels = Image.open(mask_path)  # Convert to RGB
        labels = np.array(labels)-1
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        data_samples = dict(
            labels=labels, img_path=img_path, mask_path=mask_path)
        return img, data_samples

    def __len__(self):
        return len(self.images)
