import sys
import torch as torch
from torchvision import transforms
import os
import torch.nn as nn
from model_module import *
from dataset import *
import torch.nn.functional as F 
from foundation_model import *
from preprocess_helpers import *

'''
 preprocess data for a single model (dce/dwi)

 method = 'dce' / 'dwi' which data type inputed
  
'''

def prepare_single_custom_model(method, fold, parameters, device):

  warnings.filterwarnings('ignore')
  
  batch_size = parameters['batch_size']  
  segnum = parameters['segnum']
  class_num = parameters['class_num']
  bvals =  parameters['dwi_bvals_to_use']

  channels = parameters[f'{method}_model_parameters']["channels"] 

  backbone_str = parameters  [f'{method}_model_parameters']['backbone_str']
  proj_dim = parameters[f'{method}_model_parameters']['proj_dim'] 
  enable_modality_attention = parameters[f'{method}_model_parameters']["enable_modality_attention"] 
  use_se = parameters[f'{method}_model_parameters']["use_se"] 
  
  data_key_mod = parameters['data_key_mod']
  masks_path = parameters['masks_path']
  mask_parameters =  parameters[f'{method}_model_parameters']['mask_parameters']
  mask = mask_parameters['mask']





  #variables
  channel_num=parameters[method+'_channel_num']
  key=method+str(fold)
  namelist = parameters['namelist'] #train, val, test
  adc_map = None
  nyul = None
  backbone= None
  masks_list = None 
  masks = None
  image_datasets = {}
  dataloaders_dict = {}
  train_min = None
  train_max = None
 



  #load imgs & labels
  imgs=torch.load(parameters[f'{method}_tensordata']).float()
  test_imgs=torch.load(parameters[f'{method}_test_tensordata']).float()
  labels=torch.load(parameters['labels_tensordata'])
  test_labels=torch.load(parameters['labels_test_tensordata'])
  


  # -----
  #  Prepare imgs specifically by modality, and calc and append adc map if in dwi
  # ----

  imgs, test_imgs, adc_map = prep_data_by_mod(method, bvals, imgs, test_imgs, parameters)

  # ---
  # Segment data and masks
  # ---
  if mask:
      masks = torch.load(masks_path).float()
      imgs, masks_list, labels = safe_mask_prepare(imgs, test_imgs, masks, labels, test_labels, fold, method, parameters)

  else:
    imgs,labels=data_segmentation(imgs,labels,segnum,class_num,fold)
    imgs.append(test_imgs)
    labels.append(test_labels)
    

  # --------
  # Backbone perparation
  # ---------
  if (parameters[f"{method}_model_parameters"]["backbone_str"]) is not None: 
    backbone = build_medical_backbone(parameters, device=device, method=method, in_channels=channel_num)

  #----
  # data transforms
  #----

  special_normalizer = normalize_dce_dwi(method, imgs[0], parameters, device) # use training data for normalizer

  input_size = parameters[f"{method}_model_parameters"]["input_size"]
  
  data_transforms = {
      "train": transforms.Compose([
          transforms.RandomAffine(degrees=90,translate=(0.1,0.1),shear=(0.1,0.1)),
          transforms.RandomHorizontalFlip(),
          transforms.RandomVerticalFlip(),
          transforms.Resize(input_size), # Resize images to input_size (128x128)
          special_normalizer
      ]),
      "val": transforms.Compose([
          transforms.Resize(input_size), # Resize images to input_size (128x128)
          special_normalizer
      ]),
      "test": transforms.Compose([
          transforms.Resize(input_size), # Resize images to input_size (128x128)
          special_normalizer
      ])
    }



  # ----------
  # Process datasets
  # ---------

  for i in range(len(namelist)):
    current_masks = masks_list[i] if masks_list is not None else None
    current_adc_map =  adc_map[i] if adc_map is not None else None

    image_datasets[namelist[i]] = SingleInputDataset(imgs[i],current_masks, labels[i], data_transforms[namelist[i]], modality=method, adc_map = current_adc_map)


  # ------ 
  # Create Dataloader
  # -----
  for i in range(len(namelist)):
    dataloaders_dict[namelist[i]] = torch.utils.data.DataLoader(image_datasets[namelist[i]], batch_size=batch_size, shuffle=(namelist[i] == 'train'), num_workers= 0,drop_last=False)


  # ------
  # Finally, create the model 
  # ----

  local_model=initialize_model(ModelMaskHeadBackbone(channel_num, class_num, channels, proj_dim,  enable_modality_attention=enable_modality_attention, use_se = use_se, backbone = backbone),requires_grad=True)






  #---
  # save data, after processing for reuse
  #---
  for split_idx, split in enumerate(namelist):

    dataset = image_datasets[split]

    imgs_extracted, masks_extracted, labels_extracted = extract_from_single_input_dataset(dataset)

    if method == "dwi":
        save_processed_dataset_split(
            save_path=os.path.join(parameters['data_path'], str(key+split+data_key_mod)), #methodfoldsplitdata
            imgs=imgs_extracted,
            masks=masks_extracted,
            labels=labels_extracted,
        )

    elif method == "dce":
        save_processed_dataset_split(
            save_path=os.path.join(parameters['data_path'],  str(key+split+data_key_mod)),
            imgs=imgs_extracted,
            masks=None,  #skip here already stored in dwi 
            labels=None  #skip here already stored in dwi 
        )



  return local_model, dataloaders_dict, key, labels[0], backbone





'''
  preprocess data for premade models
  
'''

def prepare_single_prebaked_model(base_path, fold, dwi_bvals= (0,1,2,3,4,5,6,7,8,9,10,11,12), dce_phases= (0,1,2,3,4,5,6)):
  pass



#----------------------- Helper functions ----------------------

def extract_from_single_input_dataset(dataset):
    all_imgs = []
    all_masks = []
    all_labels = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        masks = None
        if len(sample) == 3:
            img, masks, label = sample
        elif len(sample) == 2:
            img, label = sample
        else:
          raise ValueError(
              "invalid dataset length should be 3 or 2 but is:", len(sample) 
          )

        all_imgs.append(img.cpu())
        all_masks.append(masks.cpu() if masks is not None else None)
        all_labels.append(label if label is not None else None)

    return all_imgs, all_masks, all_labels #, all_adc

#save data set
def save_processed_dataset_split(save_path, imgs, masks, labels):
    data = {
        "imgs": imgs,              # list of tensors
        "masks": masks,            # list of tensors (duplicated OK)
        "labels": labels,          # list or tensor
    }
    torch.save(data, save_path)



#normalize dce/dwi inpu´t
def normalize_dce_dwi(method, training_imgs, parameters, debug = True, force_nyul_recalc = False):
    nyul_path = parameters["nyul_path"]
    specaial_normalizer = None

    if method == 'dwi':
      N, C, H, W = training_imgs.shape
      if parameters['dwi_add_adc_map']: 
        C-=1 #avoid last channel, it is adc map and should be normalized separately
      channel_mins = training_imgs[:, :-1].view(N, C, -1).min(dim=2)[0] 
      channel_maxs = training_imgs[:, :-1].view(N, C, -1).max(dim=2)[0]

      train_min = channel_mins.min(dim=0)[0]   # [C-1]
      train_max = channel_maxs.max(dim=0)[0]
      
      specaial_normalizer = DWINormalize(train_min, train_max)

    elif method == 'dce':

      nyul = NyulStandardizer()
              
      if debug: print('Initializing Nyul standardizer')
      if os.path.exists(nyul_path):
          try:
              nyul.load(nyul_path)
              if debug: print(f"Loaded parameters from {nyul_path}")
          except Exception as e:
              if debug: print(f"Failed to load {nyul_path}: {e}")

      if (not nyul.fitted) or force_nyul_recalc:
          if debug: print("Nyul standardizer is empty or not trained. Calculating now...")
          # Create Nyúl instance
          nyul = NyulStandardizer()
          # Fit it
          nyul.fit(training_imgs)
          if debug: print(nyul.channel_landmarks)
          # Save landmarks
          nyul.save(nyul_path)
          if debug: print("Calculation complete.")
      else:
          if debug: print("Nyul standardizer is ready to use.")

      specaial_normalizer = DCENormalize(nyul)
    return specaial_normalizer
    


#assumes a lot of things about the structure of the input data stucture careful
def prep_data_by_mod(method, bvals, imgs, test_imgs, parameters):

  if method=='dwi' and bvals is not None:


    #create normalize and append adc map  
    if parameters['dwi_add_adc_map']:

      adc_map = [None,None,None]
      #compute and normalize adc maps
      for i in range(3):
        if i == 2:  #test image exception
          adc_map[2] = compute_adc_map(test_imgs[0], bvals)
        else: 
          adc_map[i] = compute_adc_map(imgs[i], bvals)

        adc_map[i] = preprocess_adc(adc_map[i])
        
      adc_min = float(torch.min(adc_map[0])) 
      adc_max = float(torch.max(adc_map[0]))          
      for i in range(3):
          adc_map[i] = zero_to_one_adc(adc_map[i], adc_min, adc_max)
      return imgs, test_imgs, adc_map
    else:
      return imgs, test_imgs, None

  if method=='dce':
      imgs_max,_=torch.max(imgs.reshape(imgs.size(0),-1),dim=1)
      imgs=imgs/imgs_max.unsqueeze(1).unsqueeze(2).unsqueeze(3)
      test_imgs_max,_=torch.max(test_imgs.reshape(test_imgs.size(0),-1),dim=1)
      test_imgs=test_imgs/test_imgs_max.unsqueeze(1).unsqueeze(2).unsqueeze(3)

      return imgs, test_imgs, None




#prepare and append masks
def safe_mask_prepare(imgs, test_imgs, masks, labels, test_labels, fold, method, parameters):
    mask_parameters =  parameters[f'{method}_model_parameters']['mask_parameters']
    mask_target_size = mask_parameters['mask_target_size']
    segnum = parameters['segnum']
    class_num = parameters['class_num']


      # Add check for mask size before resizing
    if masks.shape[-2:] != mask_target_size:
        print(f"Resizing masks from {masks.shape[-2:]} to {mask_target_size}")
        masks = F.interpolate(masks.type(torch.FloatTensor), size=mask_target_size, mode='nearest')


    # Split masks using data_segmentation_mask
    imgs_segmented, masks_segmented, labels_segmented = data_segmentation_mask(imgs, masks, labels, segnum,class_num, fold)

    # Append test data
    imgs_segmented.append(test_imgs)
    labels_segmented.append(test_labels)
    # Append None for test masks since they don't exist
    masks_segmented.append(None)

    return imgs_segmented, masks_segmented, labels_segmented
 