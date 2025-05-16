import random
from playsound import playsound
import numpy as np
import librosa
import soundfile as sf
import time
import os
from pydub import AudioSegment
from pydub.playback import play

def arrange_swaras(swaras, swaras_per_string=8, invalid_combinations=None):
    if invalid_combinations is None:
        invalid_combinations = []

    # Function to check if a combination is invalid, considering both directions
    def is_invalid_pair(pair):
        # Check for both directions of the pair being invalid
        return pair in invalid_combinations or pair[::-1] in invalid_combinations
    
    # Function to check if the same swara repeats three times in a row
    def is_repeating_three_times(swaras_list):
        return len(swaras_list) >= 3 and swaras_list[-1] == swaras_list[-2] == swaras_list[-3]

    # List to store the resulting swaras
    arranged_swaras = ['r']  # Start with 'r' as the first swara

    # Generate swaras randomly ensuring no invalid combinations and no three consecutive same swaras
    while len(arranged_swaras) < swaras_per_string:
        new_swara = random.choice(swaras)
        
        # Skip if the combination is invalid or if the same swara appears three times in a row
        if arranged_swaras and (is_invalid_pair([arranged_swaras[-1], new_swara]) or is_repeating_three_times(arranged_swaras + [new_swara])):
            continue
        
        arranged_swaras.append(new_swara)

    # Reverse the resulting swaras before returning
    arranged_swaras = arranged_swaras[::-1]
    return arranged_swaras



# Example usage
swaras = ['s', 'r', 'g', 'p', 'n', 's#']  # Define your swaras here
invalid_combinations = [
    ['g', 's'],
    ['n', 'r'],
    ['n', 's'],
    ['n', 'g'],
    ['p', 'r'],
    ['p', 's'],
    ['s', 'n'],
    ['s', 'p'],
    ['s#', 'g'],
    ['s#', 'p'],
    ['s#', 's'],
    ['s#', 'r'],
    ['s#', 's']
]
swaras_per_string = 20  # You can change this number to adjust how many swaras you want per string

arranged = arrange_swaras(swaras, swaras_per_string, invalid_combinations)
print('Arranged swaras:', arranged)
print(' '.join(arranged))


# Hardcoded parameters
vathapi_crop = 3.7  # Crop length in seconds 2.5 for half, 3.6 for full
TARGET_DURATION = 0.3   # Seconds per swara
FADE_DURATION = 0.1     # Crossfade time
VIBRATO_FREQ = 5        # Vibrato frequency (Hz)
VIBRATO_DEPTH = 0.002   # Vibrato depth (intensity)

def load_and_adjust_duration(filename):
    if not filename == "vathapi_trimmed.wav":
        """Load a swara file and stretch it to the target duration."""
        y, sr = librosa.load(filename, sr=None)
        stretch_factor = librosa.get_duration(y=y, sr=sr) / TARGET_DURATION
        return librosa.effects.time_stretch(y, rate=stretch_factor), sr
    else:
        return librosa.load(filename, sr=None)
    
def crossfade_swaras(y1, y2, sr):
    """Apply a smooth crossfade between two swaras."""
    fade_samples = int(FADE_DURATION * sr)

    # Check if stereo (2D) or mono (1D) and reshape fade accordingly
    if y1.ndim == 2:  # Stereo case
        fade_out = np.linspace(1, 0, fade_samples).reshape(-1, 1)  # (N,1) for (N,2) audio
        fade_in = np.linspace(0, 1, fade_samples).reshape(-1, 1)
    else:  # Mono case
        fade_out = np.linspace(1, 0, fade_samples)  # Keep 1D
        fade_in = np.linspace(0, 1, fade_samples)

    y1[-fade_samples:] *= fade_out  # Now correctly applies to mono/stereo
    y2[:fade_samples] *= fade_in

    return np.concatenate((y1, y2))


def apply_vibrato(y, sr):
    """Apply vibrato effect using frequency modulation."""
    samples = np.arange(len(y))
    vibrato_wave = np.sin(2 * np.pi * VIBRATO_FREQ * samples / sr) * VIBRATO_DEPTH
    return y * (1 + vibrato_wave)

def play_and_delete(filepath):
    sound = AudioSegment.from_file(filepath)
    play(sound)  # Plays audio
    os.remove(filepath)  # Delete file after playback

def play_swara_sequence(arranged):

    # Convert swaras to filenames
    swara_files = [swara + ".wav" for swara in arranged]

    # Load and normalize the first swara
    y_final, sr = load_and_adjust_duration(swara_files[0])

    # Process the rest of the swaras
    for i in range(1, len(swara_files)):
        y_next, _ = load_and_adjust_duration(swara_files[i])
        y_final = crossfade_swaras(y_final, y_next, sr)

    # Apply vibrato
    y_final = apply_vibrato(y_final, sr)

    # Save and play the final audio
    sf.write("final_output.wav", y_final, sr)

    play_and_delete('final_output.wav')

def crop_vathapi(input_file, output_file, duration):
    # Load audio file
    y, sr = librosa.load(input_file, sr=None)  # Preserve original sample rate

    # Calculate required samples
    target_samples = int(sr * duration)

    if len(y) < target_samples:
        # Extract the last 0.5 seconds
        last_part = y[-int(sr * 0.3):]  
        y = y[:-int(sr * 0.3)]  # Remove it from the main audio

        # Stretch the last part to fill the remaining space
        remaining_samples = target_samples - len(y)
        stretch_factor = len(last_part) / remaining_samples  # Stretching ratio

        if stretch_factor < 1:  # Only stretch if there's space to fill
            last_part_stretched = librosa.effects.time_stretch(last_part, rate=stretch_factor)
        else:
            last_part_stretched = last_part  # If no stretching is needed

        # Ensure it exactly fits the remaining duration
        last_part_stretched = np.resize(last_part_stretched, remaining_samples)

        # Append the stretched last part back
        y_trimmed = np.concatenate((y, last_part_stretched))
    else:
        # Trim to specified duration
        y_trimmed = y[:target_samples]

    # Save the modified audio
    sf.write(output_file, y_trimmed, sr)

def adjust_vathapi(): 
    if vathapi_crop > 3:
        crop_vathapi('vathapi.wav', 'vathapi_trimmed.wav', vathapi_crop)
        return
    
    crop_vathapi('vathapi.wav', 'vathapi_trimmed.wav', vathapi_crop - 0.5)

    y1, sr1 = sf.read('vathapi_trimmed.wav')
    y2, sr2 = sf.read("g.wav")

    if y1.ndim != y2.ndim:
        if y1.ndim == 1:  # If y1 is mono but y2 is stereo
            y1 = np.tile(y1[:, np.newaxis], (1, 2))  # Convert mono to stereo
        elif y2.ndim == 1:  # If y2 is mono but y1 is stereo
            y2 = np.tile(y2[:, np.newaxis], (1, 2))  # Convert mono to stereo

    crossfaded_vathapi = crossfade_swaras(y1, y2, sr1)

    sf.write('vathapi_trimmed.wav', crossfaded_vathapi, sr1)

    crop_vathapi('vathapi_trimmed.wav', 'vathapi_trimmed.wav', vathapi_crop - 0.5)

    crop_vathapi('vathapi_trimmed.wav', 'vathapi_trimmed.wav', vathapi_crop)

adjust_vathapi()

for i in range(0, 10, 1):
    #playsound('vathapi_trimmed.wav')
    #arranged.append("vathapi_trimmed")
    arranged.insert(0, "vathapi_trimmed")
    play_swara_sequence(arranged)

    arranged = arrange_swaras(swaras, swaras_per_string, invalid_combinations)
    print(' '.join(arranged))

playsound('vathapi_trimmed.wav')
