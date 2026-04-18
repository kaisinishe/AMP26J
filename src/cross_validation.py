import os
import random
import numpy as np
import mir_eval
from data_loader import AMPDataLoader
from onset_detector import OnsetDetectorLFSF

def evaluate_subset(loader, detector, track_list, delta, wait):
    """Evaluates the F1 score for a specific list of tracks and parameters."""
    sum_f1 = 0.0
    tracks_processed = 0
    
    for track_id in track_list:
        try:
            track_data = loader.load_track(track_id)
            
            # Predict
            activation = detector.compute_detection_function(track_data['audio'])
            predicted_onsets, _, _ = detector.pick_peaks(
                activation, delta=delta, wait=wait, pre_avg=15, post_avg=15
            )
            
            reference_onsets = track_data['onsets']
            
            # Score
            if len(reference_onsets) > 0 and len(predicted_onsets) > 0:
                f1, _, _ = mir_eval.onset.f_measure(
                    np.array(reference_onsets), np.array(predicted_onsets), window=0.05
                )
            else:
                f1 = 0.0
                
            sum_f1 += f1
            tracks_processed += 1
        except Exception:
            pass # Skip tracks that fail to load
            
    return sum_f1 / tracks_processed if tracks_processed > 0 else 0

def k_fold_cross_validation(data_dir, k=5):
    loader = AMPDataLoader(data_dir=data_dir)
    detector = OnsetDetectorLFSF()
    all_tracks = loader.track_ids.copy()
    
    # Shuffle tracks to ensure random distribution
    random.seed(42)
    random.shuffle(all_tracks)
    
    # Create the K folds
    fold_size = len(all_tracks) // k
    folds = [all_tracks[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    
    # Handle remainders by adding them to the last fold
    if len(all_tracks) % k != 0:
        folds[-1].extend(all_tracks[k * fold_size:])
        
    validation_scores = []
    best_params_per_fold = []
    
    print(f"Starting {k}-Fold Cross Validation on {len(all_tracks)} tracks...\n")
    
    for i in range(k):
        print(f"--- FOLD {i+1}/{k} ---")
        val_tracks = folds[i]
        
        # Train tracks are all folds EXCEPT the current validation fold
        train_tracks = []
        for j in range(k):
            if j != i:
                train_tracks.extend(folds[j])
                
        # 1. Grid Search on Train Tracks
        best_f1_train = 0
        best_params = {}
        
        for test_delta in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            for test_wait in [5, 8, 10, 12]:
                f1 = evaluate_subset(loader, detector, train_tracks, test_delta, test_wait)
                if f1 > best_f1_train:
                    best_f1_train = f1
                    best_params = {'delta': test_delta, 'wait': test_wait}
                    
        print(f"  Best Params (Train): delta={best_params['delta']}, wait={best_params['wait']}")
        best_params_per_fold.append(best_params)
        
        # 2. Evaluate on Validation Tracks
        val_f1 = evaluate_subset(loader, detector, val_tracks, best_params['delta'], best_params['wait'])
        validation_scores.append(val_f1)
        print(f"  Validation F1 Score: {val_f1:.4f}\n")
        
    # Final Results
    mean_cv_score = np.mean(validation_scores)
    std_cv_score = np.std(validation_scores)
    
    print("="*40)
    print(f"CROSS VALIDATION RESULTS ({k}-Fold)")
    print(f"Expected Server F1 Score: {mean_cv_score:.4f} (±{std_cv_score:.4f})")
    print("="*40)
    
    # Return the most frequently chosen parameters to use for your final test set prediction
    return best_params_per_fold

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "train", "train") 
    k_fold_cross_validation(DATA_DIR, k=5)