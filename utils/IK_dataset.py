from PIL import Image
from torch.utils.data import Dataset
import os

class IK_Dataset(Dataset):
    def __init__(self, image_dir="" ,transform=None, target_transform=None): # define parameters
        self.image_dir = image_dir
        self.imgs = os.listdir(self.image_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_name = self.imgs[index] # read image_name
        img = Image.open(os.path.join(self.image_dir,img_name)).convert('RGB')

         # operate data transforming if defined
        if self.transform is not None:
            img = self.transform(img)  

        return {
            'image': img,
            'idx':index,
            'name':img_name,

        }


    def __len__(self):
        return len(self.imgs)   # return the length of dataset
