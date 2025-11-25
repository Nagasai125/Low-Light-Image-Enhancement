import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import os

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir=None, task=None, image_directory=None, task_type=None):
        self.image_directory = img_dir if img_dir is not None else image_directory
        self.task_type = task if task is not None else task_type
        self.image_paths = []
        self.target_paths = []

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF'}
        
        for root, dirs, filenames in os.walk(self.image_directory):
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                _, extension = os.path.splitext(filename)
                if extension in valid_extensions:
                    self.image_paths.append(os.path.join(root, filename))

        self.image_paths.sort()
        self.total_count = len(self.image_paths)
        transform_sequence = []
        transform_sequence += [transforms.ToTensor()]
        self.image_transform = transforms.Compose(transform_sequence)


    def process_image(self, filepath):

        image = Image.open(filepath).convert('RGB')
        normalized = self.image_transform(image).numpy()
        normalized = np.transpose(normalized, (1, 2, 0))
        return normalized


    def __getitem__(self, idx):

        low_light = self.process_image(self.image_paths[idx])
        low_light = np.asarray(low_light, dtype=np.float32)
        low_light = np.transpose(low_light[:, :, :], (2, 0, 1))
        filename = os.path.basename(self.image_paths[idx])

        return torch.from_numpy(low_light), filename

    def __len__(self):
        return self.total_count

DataLoader = ImageDataset
