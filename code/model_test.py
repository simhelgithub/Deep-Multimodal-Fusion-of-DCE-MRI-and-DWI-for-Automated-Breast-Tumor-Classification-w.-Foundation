
import torch as torch
import time
import copy
import torch.nn as nn
import torch.nn.functional as F

'''

def single_model_test(model, dataloaders, device,parameters):

    since = time.time()
    running_corrects = 0.
    running_mask_dice = 0.
    mask_sample_count = 0
    class0_corrects = 0.
    class1_corrects = 0.
    class2_corrects = 0.
    class3_corrects = 0.
    class0num=0.
    class1num=0.
    class2num=0.
    class3num=0.
    model.eval()
    with torch.no_grad():
        # Iterate over the test dataloader
        for batch_data in dataloaders['test']:

            # format is (inputs, labels) becasuse  there never is masks
            # Assuming test data batches contain 2 items or 3
            if len(batch_data) == 3:
                inputs, masks, labels = batch_data
            elif len(batch_data) == 2:
                inputs, labels = batch_data
                masks = None # No masks
            else:
                raise ValueError("Expected (inputs, labels) or (inputs, masks, labels) from dataloader ")


            inputs = inputs.to(device)

            if labels is not None:
                labels = labels.to(device)
            else:
              print("no lables for single model test")


            # we never have test masks
            #     masks = masks.to(device)

            # ModelMaskHead returns: classification_out, aux (dict), mask_pred (tensor)
            outputs, aux_dict, mask_output = model(inputs, masks) # Pass masks if they exist, model handles None

            # Classification accuracy calculation (only if labels exist)
            if labels is not None:
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

                class0num += torch.sum(labels.view(-1)==0)
                class1num += torch.sum(labels.view(-1)==1)
                class2num += torch.sum(labels.view(-1)==2)
                class3num += torch.sum(labels.view(-1)==3)

                class0_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==0)).item()
                class1_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==1)).item()
                class2_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==2)).item()
                class3_corrects += torch.sum((preds.view(-1) == labels.view(-1)) & (labels.view(-1)==3)).item()

        
    # Calculate epoch accuracies and print results
    num_samples = len(dataloaders['test'].dataset) if labels is not None else 0 # Use dataset size if labels exist
    # Ensure num_samples is at least the batch size if dataset size is not available
    if num_samples == 0 and len(dataloaders['test']) > 0:
        # Estimate num_samples from the last batch size if dataset size is 0
         for last_batch in dataloaders['test']: pass
         num_samples = last_batch[0].size(0) * len(dataloaders['test']) # Approximate total samples


    epoch_acc = running_corrects / num_samples if num_samples > 0 else 0

    print("test Acc: {}".format(epoch_acc))
    if labels is not None: # Only print class accuracies if labels exist
         epoch_acc0 = class0_corrects / class0num if class0num > 0 else 0
         epoch_acc1 = class1_corrects / class1num if class1num > 0 else 0
         epoch_acc2 = class2_corrects / class2num if class2num > 0 else 0
         epoch_acc3 = class3_corrects / class3num if class3num > 0 else 0
         print("test acc: [{}, {}, {}, {}]".format(epoch_acc0, epoch_acc1,epoch_acc2,epoch_acc3))


    time_elapsed = time.time() - since
    print("Testing complete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))

    #free mem
    del outputs, inputs
    torch.cuda.empty_cache()
'''

#mask _fusion always false and not tested
def fusion_model_test(dwi_model, dce_model, fusion_model, dataloaders, device, mask_fusion=False):
    """
    Test function for the  fusion model that expects raw DWI + raw DCE inputs.
    """
    since = time.time()
    fusion_model.eval()
    dwi_model.eval()
    dce_model.eval()

    running_corrects = 0
    class_corrects = [0.0, 0.0, 0.0, 0.0]
    class_counts = [0.0, 0.0, 0.0, 0.0]
    running_mask_dice = 0.0
    mask_sample_count = 0

    with torch.no_grad():
        for batch in dataloaders['test']:
            dwi_inputs, dce_inputs, masks_batch, labels = batch
            #if len(batch_data) == 3:
            #    inputs, masks, labels = batch_data
            #elif len(batch_data) == 2:
            #    inputs, labels = batch_data
            #    masks = None # No masks
            #else:
            #    raise ValueError("Expected (inputs, labels) or (inputs, masks, labels) from dataloader ")



            dwi_inputs = dwi_inputs.to(device)
            dce_inputs = dce_inputs.to(device)
            labels = labels.to(device)
            if masks_batch is not None:
                masks_batch = masks_batch.to(device)

            # First, run the individual DWI and DCE models to get their features and mask predictions
            # ModelMaskHead returns: classification_out, aux (dict), mask_pred (tensor)
            dwi_clf_out, dwi_aux, dwi_mask_pred_out = dwi_model(dwi_inputs, masks_batch)
            dce_clf_out, dce_aux, dce_mask_pred_out = dce_model(dce_inputs, masks_batch)

            # Extract the raw feature lists from the aux dictionaries
            dwi_raw_feats = dwi_aux['raw_feats']
            dce_raw_feats = dce_aux['raw_feats']

            # Forward pass on fusion model. 
            # FusionModel.forward expects: (raw_feats_dwi_list, raw_feats_dce_list, dwi_mask_pred, dce_mask_pred)
            out, fused_mask_logits, fusion_aux = fusion_model(
                dwi_raw_feats, dce_raw_feats,
                dwi_mask_pred_out, dce_mask_pred_out # Pass encoder's mask predictions
            )

            # Normalize the return into (logits, mask_outputs_or_none)
            # In FusionModel, 'out' is logits, 'fused_mask_logits' is mask_out
            logits = out
            mask_out = fused_mask_logits # Using fused_mask_logits as the mask output from fusion model

            # Classification metrics
            _, preds = torch.max(logits, dim=1)
            running_corrects += torch.sum(preds == labels).item()


            '''
            for c in range(4):
                mask_c = (labels == c)
                class_counts[c] += torch.sum(mask_c).item()
                if torch.sum(mask_c).item() > 0:
                    class_corrects[c] += torch.sum((preds == labels) & mask_c).item()

            # Mask metrics (if available)
            if masks_batch is not None and mask_out is not None:
               if mask_out.shape[-2:] != masks_batch.shape[-2:]:
                    masks_batch_resized = F.interpolate(masks_batch.float(), size=mask_out.shape[-2:], mode='nearest')
                else:
                    masks_batch_resized = masks_batch.float()

                pred_bin = (torch.sigmoid(mask_out) > 0.5).float()
                gt_bin = (masks_batch_resized > 0.5).float()

                intersection = (pred_bin * gt_bin).sum(dim=[1,2,3]).float()
                union = pred_bin.sum(dim=[1,2,3]) + gt_bin.sum(dim=[1,2,3])
                batch_dice = ((2.0 * intersection + 1e-5) / (union + 1e-5)).mean().item()

                running_mask_dice += batch_dice * dwi_inputs.size(0)
                mask_sample_count += dwi_inputs.size(0)
              '''

    # finalize metrics
    num_samples = len(dataloaders['test'].dataset)
    epoch_acc = running_corrects / num_samples if num_samples > 0 else 0.0
    class_accs = [ (class_corrects[i] / class_counts[i]) if class_counts[i] > 0 else 0.0 for i in range(4) ]
    epoch_mask_dice = running_mask_dice / mask_sample_count if mask_sample_count > 0 else 0.0
    epoch_combined = (epoch_acc + epoch_mask_dice) / 2.0 if mask_sample_count > 0 else epoch_acc

    print(f"Test Acc: {epoch_acc:.4f}")
    print(f"Per-class Acc: [{', '.join(f'{a:.4f}' for a in class_accs)}]")
    '''
    if mask_sample_count > 0:
        print(f"Mask Dice: {epoch_mask_dice:.4f}")
        print(f"Combined Score: {epoch_combined:.4f}")
    '''
    elapsed = time.time() - since
    print(f"Testing complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    # free memory
    torch.cuda.empty_cache()
