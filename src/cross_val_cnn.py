import torch
import numpy as np
import librosa
import mir_eval
from cnn_model import OnsetCNN
from onset_detector import OnsetDetectorLFSF
from data_loader import AMPDataLoader
import os
import scipy.ndimage

def evaluate_cnn_cv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Cross-Validation on: {device}")

    # 1. Setup
    DATA_DIR = os.path.join("..", "data", "train", "train")
    loader = AMPDataLoader(DATA_DIR)
    detector = OnsetDetectorLFSF()
    
    # Load the trained CNN
    model = OnsetCNN().to(device)
    model.load_state_dict(torch.load("onset_cnn_v1.pth", map_location=device))
    model.eval()

    # 2. Use a 5-fold split (20% for validation in each fold)
    tracks = loader.track_ids
    fold_size = len(tracks) // 5
    
    # Grid search ranges for CNN probabilities (0.0 - 1.0)
    delta_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    wait_range = [3, 5, 8, 10]
    
    best_global_f1 = 0
    best_global_params = {}

    for fold in range(5):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_tracks = tracks[val_start:val_end]
        
        print(f"\n--- Fold {fold+1}/5: Evaluating {len(val_tracks)} tracks ---")
        
        # Pre-calculate CNN activations for this fold to save time
        activations = []
        ground_truths = []
        
        for tid in val_tracks:
            data = loader.load_track(tid)
            spec = detector.compute_detection_function(data['audio'], return_spec=True)
            
            # Sliding Window Inference
            WINDOW_SIZE = 31
            half_window = WINDOW_SIZE // 2
            padded_spec = np.pad(spec, ((0, 0), (half_window, half_window)), mode='constant')
            
            windows = [padded_spec[:, i : i + WINDOW_SIZE] for i in range(spec.shape[1])]
            windows_tensor = torch.from_numpy(np.array(windows)).unsqueeze(1).float().to(device)
            
            probs = []
            with torch.no_grad():
                for b in range(0, len(windows_tensor), 256):
                    out = model(windows_tensor[b : b + 256])
                    probs.extend(out.squeeze().cpu().numpy())
            
            # Apply the same smoothing used in inference
            smoothed_probs = scipy.ndimage.gaussian_filter1d(np.array(probs), sigma=2.0)
            activations.append(smoothed_probs)
            ground_truths.append(data['onsets'])

        # Grid search over parameters for this fold
        best_fold_f1 = 0
        best_fold_params = {}

        for d in delta_range:
            for w in wait_range:
                f1_scores = []
                for act, gt in zip(activations, ground_truths):
                    # Peak picking
                    preds, _, _ = detector.pick_peaks(act, delta=d, wait=w)
                    
                    # Score using mir_eval (0.05s window is standard)
                    if len(preds) > 0:
                        f1 = mir_eval.onset.f_measure(gt, preds)[0]
                        f1_scores.append(f1)
                    else:
                        f1_scores.append(0)
                
                avg_f1 = np.mean(f1_scores)
                if avg_f1 > best_fold_f1:
                    best_fold_f1 = avg_f1
                    best_fold_params = {'delta': d, 'wait': w}
        
        print(f"  Best Fold F1: {best_fold_f1:.4f} with {best_fold_params}")
        
        if best_fold_f1 > best_global_f1:
            best_global_f1 = best_fold_f1
            best_global_params = best_fold_params

    print("\n" + "="*40)
    print(f"FINAL CV RESULTS")
    print(f"Best Parameters: {best_global_params}")
    print(f"Top Validation F1: {best_global_f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_cnn_cv()