import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# Create directory for spectrograms
os.makedirs("spectrograms", exist_ok=True)

# Load AIS data
df = pd.read_csv("data/ais_data.csv")

# Convert AIS feature (Speed Over Ground - SOG) into a spectrogram
for mmsi, group in df.groupby("MMSI"):
    signal = group["SOG"].values
    sr = 1  # Sample rate

    # Generate spectrogram
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Save spectrogram as an image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, cmap="inferno")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"AIS Spectrogram - MMSI {mmsi}")
    plt.savefig(f"spectrograms/{mmsi}.png")
    plt.close()

