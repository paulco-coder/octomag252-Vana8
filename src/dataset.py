import torch
from torch.utils.data import Dataset
import numpy as np

class SignalDataset(Dataset):
    def __init__(self, data_tensor, window_size=32, num_samples=1000, seed=42):
        """
        data_tensor: Tensor of shape (C, N) containing signals with NaNs.
        window_size: Size of the valid segment to extract (must be smaller than the smallest valid gap).
        num_samples: Number of artificial samples to generate per epoch.
        """
        self.data = data_tensor
        self.window_size = window_size
        self.num_samples = num_samples
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Identify valid segments (without NaNs)
        self.valid_segments = self._extract_valid_segments()

    def _extract_valid_segments(self):
        valid_segments = []
        for i in range(self.data.shape[0]):
            signal = self.data[i]
            # Find indices where signal is NOT NaN
            valid_mask = ~torch.isnan(signal)
            
            # Find consecutive True groups
            # We pad with False to easily find edges
            padded = torch.cat([torch.tensor([False]), valid_mask, torch.tensor([False])])
            diff = padded.int().diff()
            starts = torch.where(diff == 1)[0]
            ends = torch.where(diff == -1)[0]
            
            for start, end in zip(starts, ends):
                length = end - start
                if length >= self.window_size:
                    valid_segments.append((i, start.item(), end.item()))
        return valid_segments

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Randomly pick a valid segment
        if len(self.valid_segments) == 0:
            raise ValueError("No valid segments found with the given window size.")
        
        seg_idx = np.random.randint(0, len(self.valid_segments))
        c, start, end = self.valid_segments[seg_idx]
        
        # 2. Randomly crop a window
        max_start = end - self.window_size
        crop_start = np.random.randint(start, max_start + 1)
        
        ground_truth = self.data[c, crop_start : crop_start + self.window_size].clone()
        
        # 3. Generate artificial mask
        # Drop between 5% and 25% of the window
        drop_rate = np.random.uniform(0.05, 0.25)
        drop_size = max(1, int(self.window_size * drop_rate))
        
        drop_start = np.random.randint(0, self.window_size - drop_size + 1)
        
        mask = torch.ones(self.window_size)
        mask[drop_start : drop_start + drop_size] = 0.0
        
        # 4. Create masked signal (NaNs replaced by 0 for NN input)
        masked_signal = ground_truth.clone()
        masked_signal[mask == 0] = 0.0
        
        # Output: ground_truth (1, W), mask (1, W), masked_signal (1, W)
        return ground_truth.unsqueeze(0), mask.unsqueeze(0), masked_signal.unsqueeze(0)
