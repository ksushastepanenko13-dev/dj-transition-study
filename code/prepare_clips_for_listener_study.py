# prepare_clips_for_listener_study.py
import pandas as pd
import os
import shutil
import random

# Load test transitions
test_transitions = pd.read_csv('test_transitions.csv')

# === CREATE LISTENER FOLDER ===
listener_folder = 'clips_for_listeners'
os.makedirs(listener_folder, exist_ok=True)

# === RANDOMIZE ORDER (so listeners don't hear HIGH quality first) ===
random.seed(42)  # For reproducibility
randomized_order = list(range(1, len(test_transitions) + 1))
random.shuffle(randomized_order)

# === CREATE MAPPING ===
mapping = []

for listener_number, original_number in enumerate(randomized_order, 1):
    # Original file
    original_file = f'transition_clips_combo/transition_{original_number:02d}_{test_transitions.iloc[original_number-1]["pair_id"]}.mp3'

    # New simple filename for listeners
    listener_filename = f'Track_{listener_number:02d}.mp3'
    new_file = f'{listener_folder}/{listener_filename}'

    # Copy file with new name
    if os.path.exists(original_file):
        shutil.copy(original_file, new_file)

        # Store mapping
        mapping.append({
            'Listener_Track_Number': listener_number,
            'Listener_Filename': listener_filename,
            'Original_Clip_Number': original_number,
            'Pair_ID': test_transitions.iloc[original_number-1]['pair_id'],
            'Quality_Level': 'HIGH' if test_transitions.iloc[original_number-1]['smoothness_1_5'] > 3.5
                           else 'MEDIUM' if test_transitions.iloc[original_number-1]['smoothness_1_5'] >= 2.8
                           else 'LOW',
            'Predicted_Smoothness': test_transitions.iloc[original_number-1]['smoothness_1_5']
        })

        print(f"✓ Created {listener_filename} (original: transition_{original_number:02d})")

# === SAVE MASTER KEY (FOR RESEARCHER ONLY) ===
mapping_df = pd.DataFrame(mapping)
mapping_df.to_csv('listener_study_master_key.csv', index=False)

print(f"\n{'='*60}")
print(f"✓ Created {len(mapping)} tracks in '{listener_folder}/' folder")
print(f"✓ Saved master key: listener_study_master_key.csv")
print(f"{'='*60}")

print("\n📋 MASTER KEY (Keep this secret!):")
print(mapping_df.to_string(index=False))
