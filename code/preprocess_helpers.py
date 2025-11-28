
import torch as torch
import numpy as np

'''
def preprocess_dwi(dwi_tensor, parameters,
                   clip_z=(-3, 3)):    
    """
    DWI preprocessing tha

    Guarantees:
        - Stable approx mean/std before clipping
        - Preserves relative z-score 
    """

    C, H, W = dwi_tensor.shape
    out = torch.zeros_like(dwi_tensor)

    if parameters['dwi_add_adc_map']: 
      C-= 1 #don't normalize adc channel here

    z_lo, z_hi = clip_z

    for c in range(C):
        ch = dwi_tensor[c]

        # Step 1 — Z-score
        mean = ch.mean()
        std = ch.std()
        if std < 1e-6:
            std = 1e-6
        ch = (ch - mean) / std

        # Step 2 — Clip z-scores (preserves structure)
        ch = torch.clamp(ch, z_lo, z_hi)

        # Step 3 — Map to [0, 1]
        ch = (ch - z_lo) / (z_hi - z_lo)


        out[c] = ch

    return out

'''

def preprocess_dce(dce_tensor, nyul_model, apply_zscore=False):
    """
    dce_tensor: [C, H, W]
    nyul_model: fitted NyulStandardizer
    """
    C, H, W = dce_tensor.shape
    
    out = nyul_model.transform(dce_tensor, num_channels=C)
    
    # (2) Apply z-score normalization not used
    if apply_zscore:
        for c in range(C):
            mean = out[c].mean()
            std = out[c].std()
            if std > 1e-8:  # Avoid division by zero
                out[c] = (out[c] - mean) / std
    
    return torch.tensor(out, dtype=dce_tensor.dtype)



    #test
def zero_to_one_adc(adc_map, adc_min=None, adc_max=None):

    adc = (adc_map - adc_min) / (adc_max - adc_min + 1e-8)
    adc = adc.clamp(0, 1)
    return adc

def normalize_adc(adc_map):
  
    adc_clipped = adc_map.clamp(0, 3e-3)
    
    return adc_clipped / 3e-3

def preprocess_adc(adc_map):
    """
    adc_map: [1, H, W]
    """
    #adc = adc_map.clone()

    # (1) log-transform to compress outliers
    adc = torch.log1p(adc_map.clamp(min=0))

    # (2) z-score
    #mean = adc.mean()
    #std = adc.std()
    #if std < 1e-6:
    #    std = 1e-6
    #adc = (adc - mean) / std
    
    adc = normalize_adc(adc)

    return adc

#for dce
class NyulStandardizer:
    def __init__(self, landmarks=[1, 10, 25, 30, 40, 50, 60, 75, 80, 90, 99], 
                 target_range=(0,1)):  # Match DWI range
        self.landmarks = landmarks
        self.fitted = False
        self.channel_landmarks = None
        
        # Map to 0-1
        self.standard_scale = np.linspace(target_range[0], target_range[1], len(landmarks))

    def _percentiles(self, img_np):
        return np.percentile(img_np.flatten(), self.landmarks)

    def fit(self, images, num_channels=6):
        print("Fitting Nyúl standardizer...")

        all_landmarks = {c: [] for c in range(num_channels)}

        for img in images:
            if torch.is_tensor(img):
                img = img.cpu().numpy()

            for c in range(num_channels):
                all_landmarks[c].append(self._percentiles(img[c]))

        self.channel_landmarks = {
            c: np.mean(all_landmarks[c], axis=0)
            for c in range(num_channels)
        }

        self.fitted = True
        print("Nyúl fitted successfully.")

    def transform(self, img, num_channels=6):
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        is_tensor = torch.is_tensor(img)
        if is_tensor:
            img_np = img.cpu().numpy()
        else:
            img_np = img

        out = np.zeros_like(img_np, dtype=np.float32)

        for c in range(num_channels):

          x = img_np[c]


          # compute img-specific percentiles
          orig_perc = np.percentile(x.flatten(), self.landmarks)

            #  map to average landmarks
          avg_perc = self.channel_landmarks[c]

          mid = np.interp(x.flatten(), orig_perc, avg_perc)

          # map to standard scale
          mid = np.interp(mid, avg_perc, self.standard_scale)
          

          out[c] = mid.reshape(x.shape)
          
          
        if is_tensor:
            return torch.tensor(out, dtype=torch.float32)

        return out
    def save(self, path):
        np.save(path, self.channel_landmarks, self.fitted)
        print(f"Nyúl landmarks saved to: {path}")

    def load(self, path):
        self.channel_landmarks = np.load(path, allow_pickle=True).item()
        self.fitted = True
        print(f"Nyúl landmarks loaded from: {path}")


def compute_adc_map(dwi_imgs, bvals, eps=1e-6):
    """
    Computes ADC map from multi-bvalue DWI images.
    
    bvals:   list or tensor of length C

    Returns:
        adc: Tensor [1, H, W]
    """

    C, H, W = dwi_imgs.shape
    bvals = torch.tensor(bvals, dtype=torch.float32).view(C, 1, 1)

    # Avoid log(0)
    dwi_clamped = torch.clamp(dwi_imgs, min=eps)

    # log(S)
    logS = torch.log(dwi_clamped)

    # Perform linear fit log(S) = log(S0) - b * ADC
    # ADC = -slope
    # slope = Cov(b, logS) / Var(b)

    b = bvals
    mean_b = b.mean()
    mean_logS = logS.mean(dim=0)

    cov = ((b - mean_b) * (logS - mean_logS)).sum(dim=0)
    var = ((b - mean_b)**2).sum()

    slope = cov / (var + eps)
    adc = -slope

    adc = adc.unsqueeze(0)  
    return adc
    