import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import random

class OnsetDataset(Dataset):
    def __init__(self, track_ids, loader, detector, window_size=31, sample_ratio=1.0):
        self.samples = []
        self.window_size = window_size
        half_window = window_size // 2
        
        onsets_collected = []
        non_onsets_collected = []

        print(f"Slicing {len(track_ids)} tracks with LABEL WIDENING...")
        
        for track_id in track_ids:
            data = loader.load_track(track_id)
            # Get 2D Spectrogram
            spec = detector.compute_detection_function(data['audio'], return_spec=True)
            
            # Ground truth frames
            gt_frames = librosa.time_to_frames(data['onsets'], sr=detector.sr, hop_length=detector.hop_length)
            
            # Pad spec to handle edges
            padded_spec = np.pad(spec, ((0, 0), (half_window, half_window)), mode='constant')

            for i in range(spec.shape[1]):
                window = padded_spec[:, i : i + window_size]
                
                # --- WIDENING FIX ---
                # Check if current frame OR its immediate neighbors are onsets
                # This creates a 'target zone' instead of a 'target point'
                is_onset = any(neighbor in gt_frames for neighbor in range(i-1, i+2))
                label = 1.0 if is_onset else 0.0
                
                if label == 1.0:
                    onsets_collected.append((window, label))
                else:
                    non_onsets_collected.append((window, label))

        # Random Undersampling to stay balanced
        num_to_keep = int(len(onsets_collected) * sample_ratio)
        random.shuffle(non_onsets_collected)
        non_onsets_collected = non_onsets_collected[:num_to_keep]
        
        self.samples = onsets_collected + non_onsets_collected
        random.shuffle(self.samples)
        
        print(f"Dataset complete: {len(onsets_collected)} onset samples, {len(non_onsets_collected)} non-onsets.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
            window, label = self.samples[idx]
            
            # Simple Augmentation: Add tiny random noise during training
            # This makes the model more robust to "dirty" recordings
            noise = np.random.normal(0, 0.01, window.shape)
            window = window + noise
            
            return torch.from_numpy(window).unsqueeze(0).float(), torch.tensor(label).float()