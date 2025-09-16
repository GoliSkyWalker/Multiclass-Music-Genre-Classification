import librosa
import numpy as np
from scipy.signal import resample # For changing the number of samples
from magenta.models.nsynth import utils # Assuming you still use this for other parts
from magenta.models.nsynth.wavenet import fastgen
import os
# --- Configuration ---
original_audio_path = '../data/raw_audio/GTZAN/genres/blues/blues.00000.wav'  # Your 30s file
MODEL_SR = 16000  # NSynth's expected sample rate
target_duration_seconds = 30.0
target_sample_length = int(target_duration_seconds * MODEL_SR) # e.g., 64000 samples

checkpoint_path = 'wavenet-ckpt/model.ckpt-200000'
# ... (define output paths) ...

# --- Load and Speed Up Audio ---
print(f"Loading full audio from: {original_audio_path} at {MODEL_SR} Hz")
# Load the entire 30-second audio, resampled to MODEL_SR by librosa
full_audio_data, _ = librosa.load(original_audio_path, sr=MODEL_SR)
original_length_samples = full_audio_data.shape[0]
print(f"Original loaded audio: {original_length_samples} samples ({original_length_samples/MODEL_SR:.2f}s)")

print(f"Resampling to fit into {target_sample_length} samples ({target_duration_seconds}s)")
# Resample the full audio content to fit into the target_sample_length
# This effectively speeds up the audio content by a factor of:
# (original_length_samples / MODEL_SR) / target_duration_seconds
# e.g., 30s / 4s = 7.5x speed up
audio_sped_up = resample(full_audio_data, target_sample_length)
actual_sample_length = audio_sped_up.shape[0] # Should be target_sample_length

print(f"Sped-up audio: {actual_sample_length} samples, {actual_sample_length / MODEL_SR:.2f} seconds")

# --- Encode Sped-Up Audio ---
print(f"Encoding sped-up audio ({actual_sample_length} samples)...")
encoding = fastgen.encode(audio_sped_up,
                          checkpoint_path=checkpoint_path,
                          sample_length=actual_sample_length)
print(f"Encoding successful. Encoding shape: {encoding.shape}")

# --- Synthesize from Sped-Up Encoding ---
# The synthesized audio will also be `actual_sample_length` (e.g., 4 seconds)
output_dir = 'nsynth_outputs'
base_filename='blues.00000'
synthesized_audio_save_path = os.path.join(output_dir, 'genn_' + os.path.splitext(base_filename)[0] + '.wav')
print(f"Synthesizing audio ({actual_sample_length} samples) from encoding...")
fastgen.synthesize(encoding,
                      save_paths=[synthesized_audio_save_path],
                      samples_per_save=actual_sample_length,
                      checkpoint_path=checkpoint_path)
print(f"Synthesized audio saved.")