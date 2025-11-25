# rule_based_scoring.py
import pandas as pd

df = pd.read_csv('transition_features.csv')

# Define weights
TEMPO_WEIGHT = 0.35
ENERGY_WEIGHT = 0.25
TIMBRE_WEIGHT = 0.20
HARMONIC_WEIGHT = 0.20

# Calculate overall smoothness score
df['rule_based_smoothness'] = (
    df['tempo_match_score'] * TEMPO_WEIGHT +
    df['energy_match_score'] * ENERGY_WEIGHT +
    df['timbre_similarity'] * TIMBRE_WEIGHT +
    df['harmonic_score'] * HARMONIC_WEIGHT
)

# Normalize to 1-5 scale
df['smoothness_1_5'] = 1 + (df['rule_based_smoothness'] * 4)

# Save
df.to_csv('transition_scores.csv', index=False)

print(f"✓ Scored {len(df)} transitions")
print(f"\nMean smoothness: {df['smoothness_1_5'].mean():.2f}")
print(f"Range: {df['smoothness_1_5'].min():.2f} - {df['smoothness_1_5'].max():.2f}")

# Show best and worst
print("\nBest 5 transitions:")
print(df.nlargest(5, 'smoothness_1_5')[['pair_id', 'smoothness_1_5']])

print("\nWorst 5 transitions:")
print(df.nsmallest(5, 'smoothness_1_5')[['pair_id', 'smoothness_1_5']])
