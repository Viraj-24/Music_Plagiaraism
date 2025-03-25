import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def audio_to_spectrogram(audio_path, output_image):
    """Convert an audio file to a spectrogram image and save it."""
    try:
        # Load the MP3 file
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Check if the file is empty
        if len(y) == 0:
            print(f"Error: The audio file '{audio_path}' is empty or corrupted.")
            return

        # Convert to Mel spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Plot and save the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel", cmap="inferno")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mel Spectrogram: {os.path.basename(audio_path)}")
        plt.savefig(output_image, bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close()

        print(f"Spectrogram saved: {output_image}")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def process_all_audio(input_folder, output_folder):
    """Process all MP3 files in the input folder and save spectrograms in the output folder."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all MP3 files in the input folder
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

    if not audio_files:
        print("No MP3 files found in the input folder.")
        return

    for audio_file in audio_files:
        input_path = os.path.join(input_folder, audio_file)
        output_path = os.path.join(output_folder, os.path.splitext(audio_file)[0] + ".png")  # Change extension to .png

        audio_to_spectrogram(input_path, output_path)

# Example usage
process_all_audio("input", "output")
