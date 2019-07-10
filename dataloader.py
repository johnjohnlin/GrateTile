import os
import csv
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image
class myDataset(Dataset):
    def __init__(self, img_dir='/home/mediagti2/Dataset/Imagenet', train=False, transform=None):
        self.train = train
        self.img_dir = img_dir
        self.transform = transform
        self.filename_list = []
        self.folder_list = []
        self.label = []
        if self.train == True:
            self.img_dir = os.path.join(self.img_dir,'train')
        else:
            self.img_dir = os.path.join(self.img_dir,'val')
        self.folder_list = sorted(os.listdir(self.img_dir))
        for folder_name in self.folder_list:
            img_list = sorted(os.listdir(os.path.join(self.img_dir,folder_name)))
            img_list = [os.path.join(self.img_dir,folder_name,img) for img in img_list]
            self.filename_list.append(img_list)
        self.filename_list = sum(self.filename_list,[])
        for i in range(len(self.filename_list)):
            self.label.append(self.filename_list[i][37:46])
        for i in range(len(self.label)):
            self.label[i] = self.folder_list.index(self.label[i])
    def __len__(self):
        return len(self.filename_list)
    def __getitem__(self, idx):

        img = Image.open(self.filename_list[idx]).convert('RGB')
        #img = cv2.imread(self.filename_list[idx])[::-1]
        img = self.transform(img)
        label = self.label[idx]
        return img, label

def path_join(x,folder_name,root):
    return os.path.join(root,folder_name,x)

def main():
    dataset = myDataset(img_dir='/home/mediagti2/Dataset/Imagenet',
                        train=False,
                        transform=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224,224), interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                           ]))

    dataloader = Data.DataLoader(dataset, batch_size=5,
                                         shuffle=False,
                                         num_workers= 10,
                                         )
                                         
    from tqdm import tqdm
    pbar = tqdm(total=len(dataloader),ncols=120)
    for step,(img,label) in enumerate(dataloader):
        #print(step) 
        pbar.update()
        #pbar.set_postfix({'size':video.size(), 'seq':seq_len})
    pbar.close()
    
if __name__ == '__main__':
    main()
