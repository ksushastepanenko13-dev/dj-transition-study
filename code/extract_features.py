# extract_features.py
import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def extract_features(audio_file):
    """Extract all features from an audio file"""

    # Load audio (30 seconds)
    y, sr = librosa.load(audio_file, duration=30)

    # 1. TEMPO/BEAT FEATURES
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # 2. ENERGY FEATURES
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Spectral flux (energy changes)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_flux_mean = np.mean(onset_env)
    spectral_flux_std = np.std(onset_env)

    # 3. TIMBRAL FEATURES (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # Delta MFCCs (time derivatives)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfcc_delta_mean = np.mean(mfccs_delta, axis=1)

    # 4. HARMONIC FEATURES (simplified key detection)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Package all features
    features = {
        'tempo': tempo[0],  # Extract first element from array
        'beat_count': len(beats),
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'spectral_flux_mean': spectral_flux_mean,
        'spectral_flux_std': spectral_flux_std,
    }

    # Add MFCCs
    for i in range(13):
        features[f'mfcc_{i}_mean'] = mfcc_mean[i]
        features[f'mfcc_{i}_std'] = mfcc_std[i]
        features[f'mfcc_{i}_delta'] = mfcc_delta_mean[i]

    # Add chroma
    for i in range(12):
        features[f'chroma_{i}'] = chroma_mean[i]

    return features

# Main execution
if __name__ == "__main__":
    audio_dir = 'audio'
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

    print(f"Extracting features from {len(audio_files)} files...")

    all_features = []

    for audio_file in tqdm(audio_files):
        track_id = audio_file.replace('.mp3', '')
        file_path = os.path.join(audio_dir, audio_file)

        try:
            features = extract_features(file_path)
            features['track_id'] = track_id
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    # Save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv('track_features.csv', index=False)
    print(f"\n✓ Saved features for {len(df)} tracks to track_features.csv")
