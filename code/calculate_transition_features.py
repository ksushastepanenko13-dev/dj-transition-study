# calculate_transition_features.py
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# Load track features
track_features = pd.read_csv('track_features.csv')
track_features.set_index('track_id', inplace=True)

# Load transition pairs
pairs = pd.read_csv('pairs_.csv')

def calculate_transition_features(track_a_id, track_b_id):
    """Calculate features for a transition from A to B"""

    features_a = track_features.loc[track_a_id]
    features_b = track_features.loc[track_b_id]

    # 1. TEMPO MATCHING
    tempo_diff = abs(features_a['tempo'] - features_b['tempo'])
    tempo_match_score = 1.0 if tempo_diff <= 5 else max(0, 1.0 - (tempo_diff / 10))

    # 2. ENERGY ALIGNMENT
    energy_diff = abs(features_a['rms_mean'] - features_b['rms_mean'])
    energy_match_score = 1.0 / (1.0 + energy_diff * 10)

    flux_diff = abs(features_a['spectral_flux_mean'] - features_b['spectral_flux_mean'])
    flux_match_score = 1.0 / (1.0 + flux_diff)

    # 3. TIMBRE SIMILARITY
    mfcc_a = [features_a[f'mfcc_{i}_mean'] for i in range(13)]
    mfcc_b = [features_b[f'mfcc_{i}_mean'] for i in range(13)]
    timbre_distance = euclidean(mfcc_a, mfcc_b)
    timbre_similarity = 1.0 / (1.0 + timbre_distance)

    # 4. HARMONIC SIMILARITY
    chroma_a = [features_a[f'chroma_{i}'] for i in range(12)]
    chroma_b = [features_b[f'chroma_{i}'] for i in range(12)]
    chroma_corr = np.corrcoef(chroma_a, chroma_b)[0, 1]
    harmonic_score = (chroma_corr + 1) / 2

    return {
        'tempo_diff': tempo_diff,
        'tempo_match_score': tempo_match_score,
        'energy_diff': energy_diff,
        'energy_match_score': energy_match_score,
        'flux_match_score': flux_match_score,
        'timbre_distance': timbre_distance,
        'timbre_similarity': timbre_similarity,
        'harmonic_score': harmonic_score
    }

# Calculate for all pairs
transition_features = []

print(f"Calculating transition features for {len(pairs)} pairs...")

for index, row in pairs.iterrows():
    track_a_id = row['TrackA_ID']
    track_b_id = row['TrackB_ID']

    try:
        trans_features = calculate_transition_features(track_a_id, track_b_id)
        trans_features['pair_id'] = row['PairID']
        trans_features['track_a_id'] = track_a_id
        trans_features['track_b_id'] = track_b_id
        transition_features.append(trans_features)
    except Exception as e:
        print(f"✗ Error: {track_a_id} -> {track_b_id}: {e}")

# Save
df_transitions = pd.DataFrame(transition_features)
df_transitions.to_csv('transition_features.csv', index=False)

print(f"\n✓ Calculated features for {len(df_transitions)} transitions")
