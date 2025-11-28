import numpy as np
import torch as torch
import torch.nn as nn
from torch.nn import init

def data_segmentation(imgs,labels,segnum,classnum,fold):
    np.random.seed(42)
    imgs_num,channelnum,imgsize,_=imgs.shape
    totalimgs=[]
    totallabels=[]
    total_shuffled_indices=[]
    for i in range(classnum):
        subindices=torch.where(labels==i)[0]
        total_shuffled_indices.append(subindices[np.random.permutation(subindices.size(0))].tolist())
    for i in range(segnum):
        subimgs=torch.zeros(0,channelnum,imgsize,imgsize)
        sublabels=torch.zeros(0)
        for j in range(classnum):
            subnum=len(total_shuffled_indices[j])
            foldnum=int(subnum//segnum)
            if i!=(segnum-1):
                subimgs=torch.cat((subimgs,imgs[total_shuffled_indices[j][i*foldnum:(i+1)*foldnum]]),dim=0)
                sublabels=torch.cat((sublabels,labels[total_shuffled_indices[j]][i*foldnum:(i+1)*foldnum]),dim=0)
            else:
                subimgs=torch.cat((subimgs,imgs[total_shuffled_indices[j][(segnum-1)*foldnum:]]),dim=0)
                sublabels=torch.cat((sublabels,labels[total_shuffled_indices[j][(segnum-1)*foldnum:]]),dim=0)
        totalimgs.append(subimgs)
        totallabels.append(sublabels)
    train_imgs=torch.zeros(0,channelnum,imgsize,imgsize)
    train_labels=torch.zeros(0)
    for i in range(segnum):
        if i !=fold:
            train_imgs=torch.cat((train_imgs,totalimgs[i]),dim=0)
            train_labels=torch.cat((train_labels,totallabels[i]),dim=0)
        if i == fold:
            val_imgs=totalimgs[i]
            val_labels=totallabels[i]
    return [train_imgs,val_imgs],[train_labels,val_labels]

def Normalize_parameters(imgs):
    imgs=imgs.detach().numpy()
    mean=np.mean(imgs,axis=(0,2,3))
    std=np.mean(np.std(imgs,axis=(2,3)),axis=0)
    return mean.tolist(),std.tolist()

class DoubleInputDataset(torch.utils.data.Dataset):
    def __init__(self,dwi_imgs,dce_imgs,labels,dwi_transforms=None,dce_transforms=None):
        self.dwi_imgs = dwi_imgs
        self.dce_imgs = dce_imgs
        self.labels = labels
        self.dwi_transforms = dwi_transforms
        self.dce_transforms = dce_transforms
        
    def __len__(self):
        return self.labels.size(0)
    
    def __getitem__(self, index):
 
        dwi_img = self.dwi_imgs[index]
        dce_img = self.dce_imgs[index]
        lable=self.labels[index]
        if self.dwi_transforms:
            dwi_img=self.dwi_transforms(dwi_img)
        if self.dce_transforms:
            dce_img=self.dce_transforms(dce_img)
            
        return dwi_img,dce_img,lable
    
class SingleInputDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, transforms=None):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms
    def __len__(self):
        return self.labels.size(0)
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transforms:
            img = self.transforms(img)
        return img,label
    