import sys
sys.path.append(r'D:\DeepLearning\code')
import torch as torch
from torchvision import transforms
import os
import torch.nn as nn
import torch.optim as optim
import warnings
from training_function import *
from model_module import *
from dataset import *
from model_test import *

warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)
parameters=torch.load(r"D:\DeepLearning\data\parameters.pth")
model_dict=torch.load(r"D:\DeepLearning\data\model_dict.pth")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#
fold=0
reconstruct_loss=True
#
dwikey='dwi'+str(int(reconstruct_loss))+'0'+str(fold)
dcekey='dce'+str(int(reconstruct_loss))+'0'+str(fold)
finetune_dwikey='dwi'+str(int(reconstruct_loss))+'1'+str(fold)
finetune_dcekey='dce'+str(int(reconstruct_loss))+'1'+str(fold)
finetune_fusionkey='fusion'+str(int(reconstruct_loss))+'1'+str(fold)

device = parameters['device']
num_epochs = parameters['finetune_num_epochs']
batch_size = parameters['batch_size']
input_size = parameters['input_size']
segnum = parameters['segnum']
dwi_channel_num=parameters['dwi_channel_num']
dce_channel_num=parameters['dce_channel_num']

dwi_imgs=torch.load(parameters['dwi_tensordata'])
dce_imgs=torch.load(parameters['dce_tensordata'])
dwi_test_imgs=torch.load(parameters['dwi_test_tensordata'])
dce_test_imgs=torch.load(parameters['dce_test_tensordata'])

dwi_imgs=torch.log(dwi_imgs+1)
dwi_test_imgs=torch.log(dwi_test_imgs+1)

dce_imgs_max,_=torch.max(dce_imgs.reshape(dce_imgs.size(0),-1),dim=1)
dce_imgs=dce_imgs/dce_imgs_max.unsqueeze(1).unsqueeze(2).unsqueeze(3)

dce_test_imgs_max,_=torch.max(dce_test_imgs.reshape(dce_test_imgs.size(0),-1),dim=1)
dce_test_imgs=dce_test_imgs/dce_test_imgs_max.unsqueeze(1).unsqueeze(2).unsqueeze(3)

labels=torch.load(parameters['labels_tensordata'])
labels=torch.load(parameters['labels_tensordata'])

test_labels=torch.load(parameters['labels_test_tensordata'])

dwi_imgs,_=data_segmentation(dwi_imgs,labels,segnum,parameters['classnum'],fold)
dce_imgs,labels=data_segmentation(dce_imgs,labels,segnum,parameters['classnum'],fold)
dwi_mean_parameters,dwi_std_parameters=Normalize_parameters(dwi_imgs[0])
dce_mean_parameters,dce_std_parameters=Normalize_parameters(dce_imgs[0])

dwi_imgs.append(dwi_test_imgs)
dce_imgs.append(dce_test_imgs)
labels.append(test_labels)

dwi_transforms = {
    "train": transforms.Compose([
        transforms.RandomAffine(degrees=90,translate=(0.1,0.1),shear=(0.1,0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(input_size),
        transforms.Normalize(dwi_mean_parameters,dwi_std_parameters)
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize(dwi_mean_parameters,dwi_std_parameters)
    ]),
    "test": transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize(dwi_mean_parameters,dwi_std_parameters)
    ]),
}

dce_transforms = {
    "train": transforms.Compose([
        transforms.RandomAffine(degrees=90,translate=(0.1,0.1),shear=(0.1,0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(input_size),
        transforms.Normalize(dce_mean_parameters,dce_std_parameters)
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize(dce_mean_parameters,dce_std_parameters)
    ]),
    "test": transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize(dce_mean_parameters,dce_std_parameters)
    ]),
}

namelist=['train','val','test']

for i in range(3):
    dwi_imgs[i]=dwi_imgs[i].to(device)
    dce_imgs[i]=dce_imgs[i].to(device)
    labels[i]=labels[i].to(device)

image_datasets = {namelist[x]: DoubleInputDataset(dwi_imgs[x],dce_imgs[x],labels[x],dwi_transforms[namelist[x]],dce_transforms[namelist[x]]) for x in range(3)}
dataloaders_dict = {namelist[x]: torch.utils.data.DataLoader(image_datasets[namelist[x]], batch_size=batch_size, shuffle=True, num_workers= 0,drop_last=False) for x in range(3)}

dwi_model=initialize_model(model(dwi_channel_num),requires_grad=True)
dwi_model.load_state_dict(model_dict[dwikey])

dce_model=initialize_model(model(dce_channel_num),requires_grad=True)
dce_model.load_state_dict(model_dict[dcekey])

fusion_model=initialize_model(fusion_model(),requires_grad=True)

dwi_model=dwi_model.to(device)
dce_model=dce_model.to(device)
fusion_model=fusion_model.to(device)

fusion_model_optimizer=optim.Adam(fusion_model.parameters(),
                                  lr=0.001,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.001,
                                  amsgrad=False)

finetune_optimizer=optim.Adam([{'params':fusion_model.parameters()},
                             {'params':dwi_model.parameters()},
                             {'params':dce_model.parameters()}],
                              lr=0.0001,
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=0.001,
                              amsgrad=False)

criterion1 = nn.CrossEntropyLoss()
criterion2 = torch.nn.MSELoss()

dwi_model,dce_model,fusion_model,train_acc_history,train_loss_history,val_acc_history,val_loss_history= train_fushion_model(dwi_model, 
                                                                                                 dce_model, 
                                                                                                 fusion_model, 
                                                                                                 dataloaders_dict, 
                                                                                                 criterion1, 
                                                                                                 criterion2,
                                                                                                 fusion_model_optimizer, 
                                                                                                 num_epochs,
                                                                                                 reconstruct_loss,
                                                                                                 False,
                                                                                                 device)

dwi_model,dce_model,fusion_model,train_acc_history,train_loss_history,val_acc_history,val_loss_history= train_fushion_model(dwi_model, 
                                                                                                 dce_model, 
                                                                                                 fusion_model, 
                                                                                                 dataloaders_dict, 
                                                                                                 criterion1, 
                                                                                                 criterion2,
                                                                                                 finetune_optimizer, 
                                                                                                 num_epochs,
                                                                                                 reconstruct_loss,
                                                                                                 True,
                                                                                                 device)

fusion_model_test(dwi_model,dce_model,fusion_model, dataloaders_dict,device)

model_dict=torch.load(r"D:\DeepLearning\data\model_dict.pth")
model_dict[finetune_dwikey]=dwi_model.state_dict()
model_dict[finetune_dcekey]=dce_model.state_dict()
model_dict[finetune_fusionkey]=fusion_model.state_dict()
torch.save(model_dict,r"D:\DeepLearning\data\model_dict.pth")


