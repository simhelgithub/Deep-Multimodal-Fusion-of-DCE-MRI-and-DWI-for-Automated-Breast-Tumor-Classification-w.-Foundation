import torch as torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def single_model_test(model, dataloaders,device):
    model.eval()
    for phase in ["train", "val","test"]:
        total_out=torch.zeros(0,4).to(device)
        total_bi_labels=torch.zeros(0,4).to(device)
        total_labels=torch.zeros(0).to(device)
        
        running_corrects = 0.
        class0_corrects = 0.
        class1_corrects = 0.
        class2_corrects = 0.
        class3_corrects = 0.
        
        class0num=0.
        class1num=0.
        class2num=0.
        class3num=0.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.type(torch.DoubleTensor).to(device)
            labels = labels.long().to(device)
            outputs,_,features = model(inputs)
            outputs=outputs.to(device)
            _, preds = torch.max(outputs, 1)
            
            bi_labels = torch.tensor(label_binarize(labels.cpu(), classes=[0, 1, 2, 3])).to(device)
            
            total_out=torch.cat((total_out,outputs),dim=0)
            total_bi_labels=torch.cat((total_bi_labels,bi_labels),dim=0)
            total_labels=torch.cat((total_labels,labels),dim=0)
            
            class0num += torch.sum(labels.view(-1)==0)
            class1num += torch.sum(labels.view(-1)==1)
            class2num += torch.sum(labels.view(-1)==2)
            class3num += torch.sum(labels.view(-1)==3)
            running_corrects += torch.sum((preds.view(-1) == labels.view(-1)))
            class0_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==0)).item()
            class1_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==1)).item()
            class2_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==2)).item()
            class3_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==3)).item()
        epoch_acc = running_corrects / len(dataloaders[phase].dataset)
        epoch_acc0 = class0_corrects / class0num
        epoch_acc1 = class1_corrects / class1num
        epoch_acc2 = class2_corrects / class2num
        epoch_acc3 = class3_corrects / class3num
        temp_fpr_0,temp_tpr_0, temp_threshold_0 = roc_curve(total_bi_labels[:, 0].cpu(),total_out.cpu().detach().numpy()[:, 0])
        temp_fpr_1,temp_tpr_1, temp_threshold_1 = roc_curve(total_bi_labels[:, 1].cpu(),total_out.cpu().detach().numpy()[:, 1])
        temp_fpr_2,temp_tpr_2, temp_threshold_2 = roc_curve(total_bi_labels[:, 2].cpu(),total_out.cpu().detach().numpy()[:, 2])
        temp_fpr_3,temp_tpr_3, temp_threshold_3 = roc_curve(total_bi_labels[:, 3].cpu(),total_out.cpu().detach().numpy()[:, 3])
        epoch_auc0 = auc(temp_fpr_0, temp_tpr_0)
        epoch_auc1 = auc(temp_fpr_1, temp_tpr_1)
        epoch_auc2 = auc(temp_fpr_2, temp_tpr_2)
        epoch_auc3 = auc(temp_fpr_3, temp_tpr_3)
        print("{} Acc:{}".format(phase ,'%.3f' % epoch_acc))
        print("{} acc: [{}, {}, {}, {}]".format(phase, '%.2f' % epoch_acc0,  '%.2f' % epoch_acc1, '%.2f' % epoch_acc2, '%.2f' % epoch_acc3))
        print("{} auc: [{}, {}, {}, {}]".format(phase, '%.2f' % epoch_auc0,  '%.2f' % epoch_auc1, '%.2f' % epoch_auc2, '%.2f' % epoch_auc3))
        
def fusion_model_test(dwi_model, dce_model,fusion_model,dataloaders,device):
    dwi_model.eval()
    dce_model.eval()
    fusion_model.eval()
    for phase in ["train", "val","test"]:
        total_out=torch.zeros(0,4).to(device)
        total_bi_labels=torch.zeros(0,4).to(device)
        total_labels=torch.zeros(0).to(device)
        
        running_corrects = 0.
        class0_corrects = 0.
        class1_corrects = 0.
        class2_corrects = 0.
        class3_corrects = 0.
        
        class0num=0.
        class1num=0.
        class2num=0.
        class3num=0.
        for dwi_inputs,dce_inputs, labels in dataloaders[phase]:
            #inputs = inputs.type(torch.DoubleTensor).to(device)
            dwi_inputs = dwi_inputs.type(torch.DoubleTensor).to(device)
            dce_inputs = dce_inputs.type(torch.DoubleTensor).to(device)
            labels = labels.long().to(device)
            
            dwi_outputs,dwi_stackfeatures,dwi_features = dwi_model(dwi_inputs)
            dce_outputs,dce_stackfeatures,dce_features = dce_model(dce_inputs)
            
            features=torch.cat((dwi_features,dce_features),dim=1)
            outputs=fusion_model(features)
            outputs=outputs.to(device)
            _, preds = torch.max(outputs, 1)
            
            bi_labels = torch.tensor(label_binarize(labels.cpu(), classes=[0, 1, 2, 3])).to(device)
            
            total_out=torch.cat((total_out,outputs),dim=0)
            total_bi_labels=torch.cat((total_bi_labels,bi_labels),dim=0)
            total_labels=torch.cat((total_labels,labels),dim=0)
            
            class0num += torch.sum(labels.view(-1)==0)
            class1num += torch.sum(labels.view(-1)==1)
            class2num += torch.sum(labels.view(-1)==2)
            class3num += torch.sum(labels.view(-1)==3)
            running_corrects += torch.sum((preds.view(-1) == labels.view(-1)))
            class0_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==0)).item()
            class1_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==1)).item()
            class2_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==2)).item()
            class3_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==3)).item()
        epoch_acc = running_corrects / len(dataloaders[phase].dataset)
        epoch_acc0 = class0_corrects / class0num
        epoch_acc1 = class1_corrects / class1num
        epoch_acc2 = class2_corrects / class2num
        epoch_acc3 = class3_corrects / class3num
        temp_fpr_0,temp_tpr_0, temp_threshold_0 = roc_curve(total_bi_labels[:, 0].cpu(),total_out.cpu().detach().numpy()[:, 0])
        temp_fpr_1,temp_tpr_1, temp_threshold_1 = roc_curve(total_bi_labels[:, 1].cpu(),total_out.cpu().detach().numpy()[:, 1])
        temp_fpr_2,temp_tpr_2, temp_threshold_2 = roc_curve(total_bi_labels[:, 2].cpu(),total_out.cpu().detach().numpy()[:, 2])
        temp_fpr_3,temp_tpr_3, temp_threshold_3 = roc_curve(total_bi_labels[:, 3].cpu(),total_out.cpu().detach().numpy()[:, 3])
        epoch_auc0 = auc(temp_fpr_0, temp_tpr_0)
        epoch_auc1 = auc(temp_fpr_1, temp_tpr_1)
        epoch_auc2 = auc(temp_fpr_2, temp_tpr_2)
        epoch_auc3 = auc(temp_fpr_3, temp_tpr_3)
        print("{} Acc:{}".format(phase , '%.3f' % epoch_acc))
        print("{} acc: [{}, {}, {}, {}]".format(phase,'%.2f' % epoch_acc0,'%.2f' % epoch_acc1,'%.2f' % epoch_acc2,'%.2f' % epoch_acc3))
        print("{} auc: [{}, {}, {}, {}]".format(phase,'%.2f' % epoch_auc0,'%.2f' % epoch_auc1,'%.2f' % epoch_auc2,'%.2f' % epoch_auc3))
