import torch.nn as nn

class CDFRM(nn.Module):
    def __init__(self,channellist,kernel_size):
        super(CDFRM, self).__init__()
        self.channellist=channellist
        self.kernel_size=kernel_size
        self.pad_size=(self.kernel_size-1)//2
        self.conv2d = nn.Sequential()
        self.conv2d.add_module("conv1",nn.Conv2d(self.channellist[0], self.channellist[1], kernel_size=self.kernel_size, stride=1, padding=self.pad_size))
        self.conv2d.add_module("bn1",nn.BatchNorm2d(self.channellist[1]))
        self.conv2d.add_module("elu2",nn.ELU())
        self.conv2d.add_module("conv2",nn.Conv2d(self.channellist[1], self.channellist[2], kernel_size=self.kernel_size, stride=1, padding=self.pad_size))
        self.conv2d.add_module("bn2",nn.BatchNorm2d(self.channellist[2]))
        self.conv2d.add_module("elu2",nn.ELU())
        self.conv2d.add_module("conv3",nn.Conv2d(self.channellist[2], self.channellist[3], kernel_size=self.kernel_size, stride=1, padding=self.pad_size))
        self.conv2d.add_module("bn3",nn.BatchNorm2d(self.channellist[3]))
        self.conv2d.add_module("elu3",nn.ELU())
        
        self.rconv2d = nn.Sequential()
        self.rconv2d.add_module("conv1",nn.Conv2d(self.channellist[3], self.channellist[2], kernel_size=1, stride=1, padding=0))
        self.rconv2d.add_module("bn1",nn.BatchNorm2d(self.channellist[2]))
        self.rconv2d.add_module("elu1",nn.ELU())
        self.rconv2d.add_module("conv2",nn.Conv2d(self.channellist[2], self.channellist[1], kernel_size=1, stride=1, padding=0))
        self.rconv2d.add_module("bn2",nn.BatchNorm2d(self.channellist[1]))
        self.rconv2d.add_module("elu2",nn.ELU())
        self.rconv2d.add_module("conv3",nn.Conv2d(self.channellist[1], self.channellist[0], kernel_size=1, stride=1, padding=0))
        
    def forward(self, inputs):
        out=self.conv2d(inputs)
        reconstruct_inputs=self.rconv2d(out)
        return out,inputs,reconstruct_inputs


class model(nn.Module):
    def __init__(self,channel_num):
        super(model, self).__init__()
        self.channel_num=channel_num
        self.block1=CDFRM([self.channel_num,self.channel_num,self.channel_num,self.channel_num],1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.block2=CDFRM([self.channel_num,32,64,128],3)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        
        self.block3 = nn.Sequential()
        self.block3.add_module("conv1",nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True))
        self.block3.add_module("bn1",nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.block3.add_module("elu1",nn.ELU())
        self.block3.add_module("conv2",nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True))
        self.block3.add_module("bn2",nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.block3.add_module("elu2",nn.ELU())
        
        self.max = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 4)
    def forward(self, block1_input):
        block1_output,block1_input,reconstruct_block1_input=self.block1(block1_input)
        block2_input=self.maxpool1(block1_output)
        
        
        block2_output,block2_input,reconstruct_block2_input=self.block2(block2_input)
        block3_input=self.maxpool2(block2_output)
        
        block3_output=self.block3(block3_input)
        flatten_features = self.max(block3_output).view(block3_output.size(0),-1)
        out=self.fc(flatten_features)
    
        return out,[block1_input,reconstruct_block1_input,block2_input,reconstruct_block2_input],flatten_features
    
class fusion_model(nn.Module):
    def __init__(self):
        super(fusion_model, self).__init__()
        self.fc = nn.Sequential()
        self.fc.add_module("fc1",nn.Linear(1024,1024))
        self.fc.add_module("bn1",nn.BatchNorm1d(1024))
        self.fc.add_module("elu2",nn.ELU())
        self.fc.add_module("fc2",nn.Linear(1024, 1024))
        self.fc.add_module("bn2",nn.BatchNorm1d(1024))
        self.fc.add_module("elu2",nn.ELU())
        self.fc.add_module("fc3",nn.Linear(1024, 4))
    def forward(self, x):
        return self.fc(x)

    
def init_parameter(model):
    if isinstance(model, nn.Linear):
        if model.weight is not None:
            init.kaiming_uniform_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.BatchNorm1d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm3d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    else:
        pass 
            
def initialize_model(model,requires_grad):
    for param in model.parameters():
        init_parameter(param.data)
    return model