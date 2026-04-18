import os
import glob
import json
import numpy as np
import librosa
from onset_detector import OnsetDetectorLFSF

def generate_final_submission(test_data_dir, output_file):
    """
    Processes the unannotated test set and exports predictions to JSON.
    """
    # 1. Setup the Detector with your best Cross-Validation parameters
    # As found in your logs: delta=1.0, wait=5
    BEST_DELTA = 1.0
    BEST_WAIT = 5
    
    detector = OnsetDetectorLFSF()
    
    # 2. Find all .wav files in the test directory
    test_files = glob.glob(os.path.join(test_data_dir, "*.wav"))
    if not test_files:
        print(f"Error: No .wav files found in {test_data_dir}")
        return

    print(f"Found {len(test_files)} test tracks. Processing...")
    
    submission_dict = {}

    for i, wav_path in enumerate(sorted(test_files)):
        # CRITICAL: Drop the .wav extension for the JSON key 
        track_id = os.path.splitext(os.path.basename(wav_path))[0]
        
        # Load audio (standardized to 44.1kHz mono)
        audio, sr = librosa.load(wav_path, sr=44100, mono=True)
        
        # Run the detection engine
        activation = detector.compute_detection_function(audio)
        predicted_onsets, _, _ = detector.pick_peaks(
            activation, 
            delta=BEST_DELTA, 
            wait=BEST_WAIT
        )
        
        # Format the output strictly: {"TrackID": {"onsets": [list]}} 
        submission_dict[track_id] = {
            "onsets": [round(float(o), 4) for o in predicted_onsets]
        }
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_files)} tracks...")

    # 3. Write to JSON
    with open(output_file, 'w') as f:
        json.dump(submission_dict, f, indent=4)
    
    print("-" * 40)
    print(f"SUCCESS: Created {output_file}")
    print(f"Parameters used: delta={BEST_DELTA}, wait={BEST_WAIT}")
    print("-" * 40)

if __name__ == "__main__":
    # Dynamically find the data/test directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(SCRIPT_DIR, "..", "data", "test","test")
    OUTPUT_NAME = os.path.join(SCRIPT_DIR, "onset_submission.json")
    
    generate_final_submission(TEST_DIR, OUTPUT_NAME)