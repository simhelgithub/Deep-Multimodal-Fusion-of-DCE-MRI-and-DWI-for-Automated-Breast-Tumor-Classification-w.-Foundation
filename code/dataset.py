import numpy as np
import torch as torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F # Import F for resizing


#todo make handle flexible channel counts
class DWINormalize(object):
    def __init__(self, clip_z=(-3, 3), adc = True):
        self.z_lo, self.z_hi = clip_z
        self.adc = adc

    def __call__(self, img):
        # img: C,H,W
        C, H, W = img.shape
        out = torch.zeros_like(img)

        # Do NOT normalize ADC
        if self.adc: 
          C -= 1

        for ch in range(C):

            x = img[ch]

            # per-image Z-scoring
            mean = x.mean()
            std = x.std().clamp(min=1e-6)
            x = (x - mean) / std

            # clip z-scores
            x = torch.clamp(x, self.z_lo, self.z_hi)

            #map to [0,1]
            x = (x - self.z_lo) / (self.z_hi - self.z_lo)

            out[ch] = x


        return out


        
#todo make handle flexible channel counts
class DCENormalize(object):
    def __init__(self, nyul_standardizer):
        self.nyul = nyul_standardizer

    def __call__(self, img):
        # Nyul expects CPU numpy-like data
        norm = self.nyul.transform(img)
        return norm.to(img.device)


class SingleInputDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks=None, labels=None, transforms=None,
                 modality="dwi", nyul_standardizer=None, adc_min=None, adc_map=None):
        self.imgs = imgs
        self.masks = masks
        self.labels = labels
        self.transforms = transforms
        self.modality = modality
        self.adc_map = adc_map

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img = self.imgs[index].clone()  # (C,H,W)
        label = self.labels[index] if self.labels is not None else None
        mask = self.masks[index] if self.masks is not None else None
        if index == 0:
            print("DATASET SAMPLE SHAPE:", img.shape)
        if self.transforms:
            img = self.transforms(img)

        if self.adc_map is not None:
          #rescale adc map as the image has been rescaled
          adc= self.adc_map
          adc = F.interpolate(
              adc.unsqueeze(0),
              size=(img.shape[-2], img.shape[-1]),
              mode="bilinear",
              align_corners=False
          ).squeeze(0)
          img = torch.cat([img, adc], dim=0) 



        if label is not None and mask is not None:
            return img.float(), mask.float(), label
        elif label is not None:
            return img.float(), label
        elif mask is not None:
            return img.float(), mask.float()
        return img.float()
        
class LoadedFusionDataset(torch.utils.data.Dataset):
    """
    Takes already stored and preprocessed DWI / DCE / masks / labels
    returns:
        (dwi, dce, mask?, label?)
    """

    def __init__(self, dwi, dce, masks=None, labels=None):
        self.dwi = dwi
        self.dce = dce
        self.masks = masks
        self.labels = labels

        self.length = len(dwi)
        assert len(dwi) == len(dce), "DWI and DCE must have same length"
        if masks is not None:
            assert len(masks) == len(dwi), "Masks must match DWI length"
        if labels is not None:
            assert len(labels) == len(dwi), "Labels must match DWI length"

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x1 = self.dwi[index].clone()
        x2 = self.dce[index].clone()

        m = self.masks[index] if self.masks is not None else None
        y = self.labels[index] if self.labels is not None else None
        if m is not None and y is not None:
            return x1.float(), x2.float(), m.float(), y

        if m is not None:
            return x1.float(), x2.float(), m.float()

        if y is not None:
            return x1.float(), x2.float(), y

        return x1.float(), x2.float()



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



def data_segmentation_mask(imgs, masks, labels, segnum, classnum, fold):


    np.random.seed(42)
    imgs_num, channelnum, img_height, img_width = imgs.shape
    masks_num, mask_channelnum, mask_height, mask_width = masks.shape


    totalimgs=[]
    totallabels=[]
    totalmasks = [] 
    total_shuffled_indices=[]
    for i in range(classnum):
        subindices=torch.where(labels==i)[0]
        total_shuffled_indices.append(subindices[np.random.permutation(subindices.size(0))].tolist())

    for i in range(segnum):
        subimgs=torch.zeros(0,channelnum,img_height,img_width, dtype=imgs.dtype)
        submasks=torch.zeros(0,mask_channelnum,mask_height,mask_width, dtype=masks.dtype) #  submasks tensor with the correct dimensions
        sublabels=torch.zeros(0, dtype=labels.dtype) # Initialize sublabels tensor


        for j in range(classnum):
            subnum=len(total_shuffled_indices[j])
            foldnum=int(subnum//segnum)
            start_idx = total_shuffled_indices[j][i*foldnum:(i+1)*foldnum] if i != (segnum-1) else total_shuffled_indices[j][(segnum-1)*foldnum:]

            # Get image and mask subsets
            img_subset = imgs[start_idx]
            mask_subset = masks[start_idx]
            label_subset = labels[total_shuffled_indices[j]][i*foldnum:(i+1)*foldnum] if i != (segnum-1) else labels[total_shuffled_indices[j]][(segnum-1)*foldnum:]


            # No need to resize masks here, as they are already resized before being passed in

            subimgs=torch.cat((subimgs,img_subset),dim=0) 
            submasks=torch.cat((submasks,mask_subset),dim=0) # Concatenate mask_subset directly
            sublabels=torch.cat((sublabels,label_subset),dim=0)

        totalimgs.append(subimgs)
        totalmasks.append(submasks) # Append segmented masks
        totallabels.append(sublabels)

    train_imgs=torch.zeros(0,channelnum,img_height,img_width, dtype=imgs.dtype)
    train_masks=torch.zeros(0,mask_channelnum,mask_height,mask_width, dtype=masks.dtype) # Initialize train_masks tensor
    train_labels=torch.zeros(0, dtype=labels.dtype)

    for i in range(segnum):
        if i !=fold:
            train_imgs=torch.cat((train_imgs,totalimgs[i]),dim=0)
            train_masks=torch.cat((train_masks,totalmasks[i]),dim=0) # Concatenate train masks
            train_labels=torch.cat((train_labels,totallabels[i]),dim=0)
        if i == fold:
            val_imgs=totalimgs[i]
            val_masks=totalmasks[i] # Assign validation masks
            val_labels=totallabels[i]

    return [train_imgs,val_imgs],[train_masks, val_masks],[train_labels,val_labels] # Return segmented masks

def Normalize_parameters(imgs):
    print(f"Normalize_parameters: Input imgs shape: {imgs.shape}") 
    imgs=imgs.detach().numpy()
    mean=np.mean(imgs,axis=(0,2,3))
    std=np.mean(np.std(imgs,axis=(2,3)),axis=0)
    print(f"Normalize_parameters: Output mean length: {len(mean.tolist())}") 
    return mean.tolist(),std.tolist()

