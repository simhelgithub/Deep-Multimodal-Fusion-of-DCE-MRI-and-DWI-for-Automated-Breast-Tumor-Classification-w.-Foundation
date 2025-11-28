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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#
method='dwi'
fold=0
reconstruct_loss=True
#
fushion_finetune=False
key=method+str(int(reconstruct_loss))+str(int(fushion_finetune))+str(fold)

device = parameters['device']
num_epochs = parameters['num_epochs']
batch_size = parameters['batch_size']
input_size = parameters['input_size']
segnum = parameters['segnum']
channel_num=parameters[method+'_channel_num']

imgs=torch.load(parameters[method+'_tensordata'])
test_imgs=torch.load(parameters[method+'_test_tensordata'])

if method=='dwi':
    imgs=torch.log(imgs+1)
    test_imgs=torch.log(test_imgs+1)
if method=='dce':
    imgs_max,_=torch.max(imgs.reshape(imgs.size(0),-1),dim=1)
    imgs=imgs/imgs_max.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    test_imgs_max,_=torch.max(test_imgs.reshape(test_imgs.size(0),-1),dim=1)
    test_imgs=test_imgs/test_imgs_max.unsqueeze(1).unsqueeze(2).unsqueeze(3)

labels=torch.load(parameters['labels_tensordata'])
test_labels=torch.load(parameters['labels_test_tensordata'])
imgs,labels=data_segmentation(imgs,labels,segnum,parameters['classnum'],fold)

mean_parameters,std_parameters=Normalize_parameters(imgs[0])

imgs.append(test_imgs)
labels.append(test_labels)

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomAffine(degrees=90,translate=(0.1,0.1),shear=(0.1,0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(input_size),
        transforms.Normalize(mean_parameters,std_parameters)
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize(mean_parameters,std_parameters)
    ]),
    "test": transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize(mean_parameters,std_parameters)
    ])
}

namelist=['train','val','test']

for i in range(3):
    imgs[i]=imgs[i].to(device)
    labels[i]=labels[i].to(device)

image_datasets = {namelist[x]: SingleInputDataset(imgs[x],labels[x],data_transforms[namelist[x]]) for x in range(3)}
dataloaders_dict = {namelist[x]: torch.utils.data.DataLoader(image_datasets[namelist[x]], batch_size=batch_size, shuffle=True, num_workers= 0,drop_last=False) for x in range(3)}

model=initialize_model(model(channel_num),requires_grad=True)

model=model.to(device)

optimizer=optim.Adam(model.parameters(),
                             lr=0.001,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0.0001,
                             amsgrad=False)

criterion1 = nn.CrossEntropyLoss()
criterion2 = torch.nn.MSELoss()

model,train_acc_history,train_loss_history,val_acc_history,val_loss_history= train_model(model, 
                                                                                         dataloaders_dict, 
                                                                                         criterion1, 
                                                                                         criterion2,
                                                                                         optimizer, 
                                                                                         num_epochs,
                                                                                         reconstruct_loss,
                                                                                         device)

single_model_test(model, dataloaders_dict,parameters['device'])

model_dict=torch.load(r"D:\DeepLearning\data\model_dict.pth")
model_dict[key]=model.state_dict()
torch.save(model_dict,r"D:\DeepLearning\data\model_dict.pth")
