import torch
import numpy as np
import librosa
import scipy.ndimage
import matplotlib.pyplot as plt
from cnn_model import OnsetCNN
from onset_detector import OnsetDetectorLFSF
from data_loader import AMPDataLoader
import os

def run_cnn_inference(track_id, model_path="onset_cnn_v1.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = OnsetCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Setup Data
    DATA_DIR = os.path.join("..", "data", "train", "train")
    loader = AMPDataLoader(DATA_DIR)
    track_data = loader.load_track(track_id)
    detector = OnsetDetectorLFSF()
    
    # Get spectrogram for inference
    spec = detector.compute_detection_function(track_data['audio'], return_spec=True)
    
    # 3. Batch Inference (Sliding Window)
    window_size = 31
    half_window = window_size // 2
    padded_spec = np.pad(spec, ((0, 0), (half_window, half_window)), mode='constant')
    
    windows = [padded_spec[:, i : i + window_size] for i in range(spec.shape[1])]
    windows_tensor = torch.from_numpy(np.array(windows)).unsqueeze(1).float().to(device)
    
    probabilities = []
    with torch.no_grad():
        for i in range(0, len(windows_tensor), 256): # Batch size of 256 for speed
            batch = windows_tensor[i : i + 256]
            output = model(batch)
            probabilities.extend(output.squeeze().cpu().numpy())
    
    # --- SMOOTHING FIX ---
    cnn_activation = np.array(probabilities)
    # Apply Gaussian smoothing to eliminate the jitter/barcode look
    cnn_activation = scipy.ndimage.gaussian_filter1d(cnn_activation, sigma=2.0)
    
    # 4. Peak Picking
    # Since smoothed probabilities are clean, we use a lower delta (try 0.1 to 0.3)
    predicted_onsets, _, _ = detector.pick_peaks(cnn_activation, delta=0.15, wait=5)
    
    # 5. Visualization
    time_frames = librosa.frames_to_time(np.arange(len(cnn_activation)), sr=detector.sr, hop_length=detector.hop_length)
    time_audio = np.linspace(0, track_data['duration_sec'], len(track_data['audio']))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    ax1.plot(time_frames, cnn_activation, color='blue', label="CNN Smoothed Probability")
    ax1.set_title(f"CNN Activation (Smoothed): {track_id}")
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    
    ax2.plot(time_audio, np.abs(track_data['audio']), color='gray', alpha=0.4, label="Waveform")
    ax2.vlines(track_data['onsets'], 0, 1, color='red', linestyle='--', label='Ground Truth')
    ax2.vlines(predicted_onsets, 0, 0.8, color='blue', label='CNN Predictions')
    ax2.set_title("Detection Comparison")
    ax2.set_xlim(0, 10)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test on the same track as before to see the improvement!
    run_cnn_inference("ff123_2nd_vent_clip")