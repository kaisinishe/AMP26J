import os
import torch
import numpy as np
import librosa
import scipy.ndimage
from cnn_model import OnsetCNN
from onset_detector import OnsetDetectorLFSF
from data_loader import AMPDataLoader

def export_all_activations(model_path="onset_cnn_v1.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join("..", "data", "activations")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Setup
    model = OnsetCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    loader = AMPDataLoader(os.path.join("..", "data", "train", "train"))
    detector = OnsetDetectorLFSF()
    
    print(f"Exporting activations to {output_dir}...")

    for i, tid in enumerate(loader.track_ids):
        data = loader.load_track(tid)
        spec = detector.compute_detection_function(data['audio'], return_spec=True)
        
        # Batch Inference
        WINDOW_SIZE = 31
        half_window = WINDOW_SIZE // 2
        padded_spec = np.pad(spec, ((0, 0), (half_window, half_window)), mode='constant')
        windows = [padded_spec[:, j : j + WINDOW_SIZE] for j in range(spec.shape[1])]
        windows_tensor = torch.from_numpy(np.array(windows)).unsqueeze(1).float().to(device)
        
        probs = []
        with torch.no_grad():
            for b in range(0, len(windows_tensor), 512):
                out = model(windows_tensor[b : b + 512])
                probs.extend(out.squeeze().cpu().numpy())
        
        # Smooth
        activation = scipy.ndimage.gaussian_filter1d(np.array(probs), sigma=2.0)
        
        # Save as .npy
        np.save(os.path.join(output_dir, f"{tid}_act.npy"), activation)
        
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(loader.track_ids)}] Exported {tid}")

if __name__ == "__main__":
    export_all_activations()