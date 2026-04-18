import os
import glob
import json
import torch
import numpy as np
import librosa
import scipy.ndimage
from cnn_model import OnsetCNN
from onset_detector import OnsetDetectorLFSF

def generate_cnn_submission(test_data_dir, output_file, model_path="onset_cnn_v1.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission using CNN on: {device}")

    # 1. Setup Models and Parameters
    # Using your CHAMPION parameters from CV
    BEST_DELTA = 0.15
    BEST_WAIT = 3
    WINDOW_SIZE = 31
    
    model = OnsetCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    detector = OnsetDetectorLFSF()
    
    test_files = glob.glob(os.path.join(test_data_dir, "*.wav"))
    if not test_files:
        print(f"Error: No .wav files found in {test_data_dir}")
        return

    submission_dict = {}

    for i, wav_path in enumerate(sorted(test_files)):
        track_id = os.path.splitext(os.path.basename(wav_path))[0]
        audio, _ = librosa.load(wav_path, sr=44100, mono=True)
        
        # Get Spectrogram
        spec = detector.compute_detection_function(audio, return_spec=True)
        
        # Batch Inference
        half_window = WINDOW_SIZE // 2
        padded_spec = np.pad(spec, ((0, 0), (half_window, half_window)), mode='constant')
        windows = [padded_spec[:, j : j + WINDOW_SIZE] for j in range(spec.shape[1])]
        windows_tensor = torch.from_numpy(np.array(windows)).unsqueeze(1).float().to(device)
        
        probs = []
        with torch.no_grad():
            for b in range(0, len(windows_tensor), 256):
                out = model(windows_tensor[b : b + 256])
                probs.extend(out.squeeze().cpu().numpy())
        
        # Smooth and Pick Peaks
        cnn_activation = scipy.ndimage.gaussian_filter1d(np.array(probs), sigma=2.0)
        predicted_onsets, _, _ = detector.pick_peaks(
            cnn_activation, 
            delta=BEST_DELTA, 
            wait=BEST_WAIT
        )
        
        submission_dict[track_id] = {
            "onsets": [round(float(o), 4) for o in predicted_onsets]
        }
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_files)} tracks...")

    with open(output_file, 'w') as f:
        json.dump(submission_dict, f, indent=4)
    
    print("-" * 40)
    print(f"SUCCESS: Created {output_file}")
    print(f"Final Params: delta={BEST_DELTA}, wait={BEST_WAIT}")
    print("-" * 40)

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Adjust this path if your test files are in data/test/test
    TEST_DIR = os.path.join(SCRIPT_DIR, "..", "data", "test", "test")
    OUTPUT_NAME = os.path.join(SCRIPT_DIR, "onset_submission_cnn.json")
    
    generate_cnn_submission(TEST_DIR, OUTPUT_NAME)