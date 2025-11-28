import torch as torch
import time
import copy

def train_model(model, dataloaders, criterion1,criterion2, optimizer, num_epochs,reconstruct_loss,device):
    since = time.time()
    train_acc_history = [] 
    train_loss_history = [] 
    val_acc_history = [] 
    val_loss_history = [] 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.
    for epoch in range(num_epochs):
        print("="*20)
        print("Epoch {}/{}".format(epoch, num_epochs-1))

        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            class0_corrects = 0.
            class1_corrects = 0.
            class2_corrects = 0.
            class3_corrects = 0.
            class0num=0.
            class1num=0.
            class2num=0.
            class3num=0.
            if phase=="train":
                model.train()

            else:
                model.eval()
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.type(torch.DoubleTensor).to(device)
                labels = labels.long().to(device)
                
                with torch.autograd.set_grad_enabled(phase=="train"):
                    outputs,stackfeatures,features = model(inputs)
                    
                    loss= criterion1(outputs,labels)
                    
                    if reconstruct_loss:
                        reconstruct_loss=criterion2(stackfeatures[0],stackfeatures[1])+criterion2(stackfeatures[2],stackfeatures[3])
                        loss=loss+reconstruct_loss
                _, preds = torch.max(outputs, 1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
                
                class0num += torch.sum(labels.view(-1)==0)
                class1num += torch.sum(labels.view(-1)==1)
                class2num += torch.sum(labels.view(-1)==2)
                class3num += torch.sum(labels.view(-1)==3)
                
                class0_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==0)).item()
                class1_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==1)).item()
                class2_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==2)).item()
                class3_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==3)).item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_acc0 = class0_corrects / class0num
            epoch_acc1 = class1_corrects / class1num
            epoch_acc2 = class2_corrects / class2num
            epoch_acc3 = class3_corrects / class3num
       
            print("{} Acc:{} Loss: {}".format(phase ,epoch_acc, epoch_loss ))
            print("{} acc: [{}, {}, {}, {}]".format(phase,epoch_acc0, epoch_acc1,epoch_acc2,epoch_acc3))
            print('-'*15)
            if phase == "val" and epoch_acc >= best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == "val":
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
    
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_val_acc))
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    model.load_state_dict(best_model_wts)

    return model,train_acc_history,train_loss_history,val_acc_history,val_loss_history

def train_fushion_model(dwi_model,dce_model,fusion_model, dataloaders, criterion1,criterion2, optimizer, num_epochs,reconstruct_loss,finetune,device):
    since = time.time()
    train_acc_history = [] 
    train_loss_history = [] 
    val_acc_history = [] 
    val_loss_history = [] 
    best_dwi_model_wts = copy.deepcopy(dwi_model.state_dict())
    best_dce_model_wts = copy.deepcopy(dce_model.state_dict())
    best_fusion_model_wts = copy.deepcopy(fusion_model.state_dict())
    best_val_acc = 0.
    for epoch in range(num_epochs):
        print("="*20)
        print("Epoch {}/{}".format(epoch, num_epochs-1))

        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            class0_corrects = 0.
            class1_corrects = 0.
            class2_corrects = 0.
            class3_corrects = 0.
            
            class0num=0.
            class1num=0.
            class2num=0.
            class3num=0.
            
            if finetune:
                if phase=="train":
                    dwi_model.train()
                    dce_model.train()
                    fusion_model.train()
                else:
                    dwi_model.eval()
                    dce_model.eval()
                    fusion_model.eval()
            else:
                dwi_model.eval()
                dce_model.eval()
                if phase=="train":
                    fusion_model.train()
                else:
                    fusion_model.eval()
            if phase == "train":
                fusion_model.train()
            else: 
                dwi_model.eval()
                dce_model.eval()
                fusion_model.eval()
            
            for dwi_inputs,dce_inputs, labels in dataloaders[phase]:
                dwi_inputs = dwi_inputs.type(torch.DoubleTensor).to(device)
                dce_inputs = dce_inputs.type(torch.DoubleTensor).to(device)
                labels = labels.long().to(device)
                
                with torch.autograd.set_grad_enabled(phase=="train"):
                    dwi_outputs,dwi_stackfeatures,dwi_features = dwi_model(dwi_inputs)
                    dce_outputs,dce_stackfeatures,dce_features = dce_model(dce_inputs)
                    
                    features=torch.cat((dwi_features,dce_features),dim=1)
                    fushion_outputs=fusion_model(features)
                    
                    loss= criterion1(fushion_outputs,labels)
                    if finetune:
                        if reconstruct_loss:
                            dwi_reconstruct_loss=criterion2(dwi_stackfeatures[0],dwi_stackfeatures[1])+criterion2(dwi_stackfeatures[2],dwi_stackfeatures[3])
                            dce_reconstruct_loss=criterion2(dce_stackfeatures[0],dce_stackfeatures[1])+criterion2(dce_stackfeatures[2],dce_stackfeatures[3])
                            loss+=(dwi_reconstruct_loss+dce_reconstruct_loss)
                _, preds = torch.max(fushion_outputs, 1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
                
                class0num += torch.sum(labels.view(-1)==0)
                class1num += torch.sum(labels.view(-1)==1)
                class2num += torch.sum(labels.view(-1)==2)
                class3num += torch.sum(labels.view(-1)==3)
                
                class0_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==0)).item()
                class1_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==1)).item()
                class2_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==2)).item()
                class3_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==3)).item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_acc0 = class0_corrects / class0num
            epoch_acc1 = class1_corrects / class1num
            epoch_acc2 = class2_corrects / class2num
            epoch_acc3 = class3_corrects / class3num
       
            print("{} Acc:{} Loss: {}".format(phase ,epoch_acc, epoch_loss ))
            print("{} acc: [{}, {}, {}, {}]".format(phase,epoch_acc0, epoch_acc1,epoch_acc2,epoch_acc3))
            print('-'*15)
            if phase == "val" and epoch_acc >= best_val_acc:
                best_val_acc = epoch_acc
                best_dwi_model_wts = copy.deepcopy(dwi_model.state_dict())
                best_dce_model_wts = copy.deepcopy(dce_model.state_dict())
                best_fusion_model_wts = copy.deepcopy(fusion_model.state_dict())
                
            if phase == "val":
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
    
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_val_acc))
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    dwi_model.load_state_dict(best_dwi_model_wts)
    dce_model.load_state_dict(best_dce_model_wts)
    fusion_model.load_state_dict(best_fusion_model_wts)