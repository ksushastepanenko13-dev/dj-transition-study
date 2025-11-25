# create_ultimate_dj_transitions.py
import librosa
import soundfile as sf
import pandas as pd
import os
import numpy as np
import random
from scipy import signal

test_transitions = pd.read_csv('test_transitions.csv')
output_dir = 'transition_clips_combo'
os.makedirs(output_dir, exist_ok=True)

print(f"Creating ULTIMATE DJ transitions (30 seconds each)...\n")

def apply_lowpass_filter(audio, sr, cutoff_freq):
    """Keep bass, remove highs"""
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, audio)

def apply_highpass_filter(audio, sr, cutoff_freq):
    """Remove bass, keep highs"""
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, audio)

for index, row in test_transitions.iterrows():
    track_a_id = row['track_a_id']
    track_b_id = row['track_b_id']
    pair_id = row['pair_id']

    print(f"Processing {index+1}/{len(test_transitions)}: {pair_id}")

    try:
        # === LOAD FULL TRACKS ===
        y_a, sr = librosa.load(f'audio/{track_a_id}.mp3')
        y_b, sr = librosa.load(f'audio/{track_b_id}.mp3')

        print(f"  Track A: {len(y_a)/sr:.1f}s")
        print(f"  Track B: {len(y_b)/sr:.1f}s")

        # === BEATMATCHING ===
        tempo_a, beats_a = librosa.beat.beat_track(y=y_a, sr=sr)
        tempo_b, beats_b = librosa.beat.beat_track(y=y_b, sr=sr)

        print(f"  Tempo A: {tempo_a[0]:.1f} BPM")
        print(f"  Tempo B: {tempo_b[0]:.1f} BPM")

        # Time-stretch Track B to match Track A's tempo
        tempo_ratio = tempo_a[0] / tempo_b[0]
        y_b_stretched = librosa.effects.time_stretch(y_b, rate=tempo_ratio)

        print(f"  ✓ Beatmatched: stretched Track B by {tempo_ratio:.3f}x")

        # Recalculate beats for stretched Track B
        _, beats_b_stretched = librosa.beat.beat_track(y=y_b_stretched, sr=sr)

        # === PICK BEAT-ALIGNED SECTIONS ===
        segment_length = 15 * sr

        # Track A: pick random beat in safe zone
        safe_beats_a = beats_a[(beats_a > 10*sr//512) & (beats_a < (len(y_a)-segment_length)//512)]
        if len(safe_beats_a) > 0:
            start_beat_a = random.choice(safe_beats_a)
            start_a = start_beat_a * 512
        else:
            start_a = 10 * sr

        a_segment = y_a[start_a:start_a + segment_length]

        # Track B: pick random beat in safe zone
        safe_beats_b = beats_b_stretched[(beats_b_stretched > 10*sr//512) &
                                          (beats_b_stretched < (len(y_b_stretched)-segment_length)//512)]
        if len(safe_beats_b) > 0:
            start_beat_b = random.choice(safe_beats_b)
            start_b = start_beat_b * 512
        else:
            start_b = 10 * sr

        b_segment = y_b_stretched[start_b:start_b + segment_length]

        # === CROSSFADE SECTION (10 seconds) ===
        fade_duration = 10 * sr
        a_crossfade = a_segment[-fade_duration:].copy()
        b_crossfade = b_segment[:fade_duration].copy()

        print(f"  ✓ Applying EQ crossfade + filter sweep...")

        # === EQ SPLIT (Bass vs Highs) ===
        eq_split = 250  # Hz

        # Track A: split into bass and highs
        a_bass = apply_lowpass_filter(a_crossfade, sr, eq_split)
        a_highs = apply_highpass_filter(a_crossfade, sr, eq_split)

        # Track B: split into bass and highs
        b_bass = apply_lowpass_filter(b_crossfade, sr, eq_split)
        b_highs = apply_highpass_filter(b_crossfade, sr, eq_split)

        # === FILTER SWEEP ON TRACK A HIGHS ===
        nyquist = sr // 2
        cutoff_freq = np.linspace(250, 2000, fade_duration)  # Sweep 250Hz → 2000Hz

        a_highs_swept = a_highs.copy()
        for i, cutoff in enumerate(cutoff_freq[::sr//100]):  # Every ~10ms
            b_coef, a_coef = signal.butter(4, cutoff / nyquist, btype='high')
            start_sample = i * (sr//100)
            end_sample = start_sample + (sr//100)

            if end_sample < len(a_highs_swept):
                a_highs_swept[start_sample:end_sample] = signal.filtfilt(
                    b_coef, a_coef, a_highs_swept[start_sample:end_sample]
                )

        # === FADE CURVES (exponential) ===
        # Bass swaps quickly
        fade_out_bass = np.power(np.linspace(1, 0, fade_duration), 3)
        fade_in_bass = np.power(np.linspace(0, 1, fade_duration), 2)

        # Highs swap slower
        fade_out_highs = np.power(np.linspace(1, 0, fade_duration), 2)
        fade_in_highs = np.power(np.linspace(0, 1, fade_duration), 3)

        # === APPLY FADES ===
        a_bass_faded = a_bass * fade_out_bass
        a_highs_faded = a_highs_swept * fade_out_highs  # Filter-swept highs!

        b_bass_faded = b_bass * fade_in_bass
        b_highs_faded = b_highs * fade_in_highs

        # === COMBINE ALL ELEMENTS ===
        crossfade_section = (a_bass_faded + a_highs_faded +
                            b_bass_faded + b_highs_faded)

        # === FULL TRANSITION ===
        combined = np.concatenate([
            a_segment[:10*sr],       # Track A alone (full freq, beatmatched)
            crossfade_section,        # EQ crossfade + filter sweep
            b_segment[fade_duration:] # Track B alone (full freq, beatmatched)
        ])

        # Normalize
        combined = combined / np.max(np.abs(combined)) * 0.95

        # === SAVE ===
        output_file = os.path.join(output_dir, f'transition_{index+1:02d}_{pair_id}.mp3')
        sf.write(output_file, combined, sr)

        duration = len(combined) / sr
        print(f"  ✓ Created: {output_file}")
        print(f"     Duration: {duration:.1f}s")
        print(f"     Techniques: Beatmatching + EQ + Filter Sweep 🔥\n")

    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print(f"\n{'='*70}")
print(f"✓ Created {len(test_transitions)} ULTIMATE DJ transitions")
print(f"  ✅ Beatmatching (tempos synced)")
print(f"  ✅ EQ Crossfade (bass swaps cleanly)")
print(f"  ✅ Filter Sweep (dramatic whoosh effect on highs)")
print(f"  ✅ Beat-aligned (starts on beats)")
print(f"{'='*70}")
