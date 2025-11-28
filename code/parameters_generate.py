import torch as torch

parameters={}

parameters['dwi_channel_num']=13
parameters['dce_channel_num']=6

parameters['dwi_tensordata']=r'D:\DeepLearning\data\dwi_tensordata.pth'
parameters['dce_tensordata']=r'D:\DeepLearning\data\dce_tensordata.pth'
parameters['labels_tensordata']=r'D:\DeepLearning\data\labels_tensordata.pth'

parameters['dwi_test_tensordata']=r'D:\DeepLearning\data\dwi_test_tensordata.pth'
parameters['dce_test_tensordata']=r'D:\DeepLearning\data\dce_test_tensordata.pth'
parameters['labels_test_tensordata']=r'D:\DeepLearning\data\labels_test_tensordata.pth'

parameters['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parameters['num_epochs']=1500
parameters['finetune_num_epochs']=20
parameters['batch_size']=32
parameters['input_size']=32
parameters['segnum']=5
parameters['classnum']=4

torch.save(parameters,r"D:\DeepLearning\data\parameters.pth")

model_dict={}
torch.save(model_dict,r"D:\DeepLearning\data\model_dict.pth")
fusion_model_dict={}
torch.save(fusion_model_dict,r"D:\DeepLearning\data\fusion_model_dict.pth")