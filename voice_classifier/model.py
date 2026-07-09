"""Small CNN that classifies a voice clip as HUMAN or AI.

Input: log-mel spectrogram of a fixed-length window (default 3 s @ 16 kHz).
Device-agnostic: runs on CUDA, ROCm (also exposed as torch.cuda) and CPU.
"""

import torch
import torch.nn as nn

SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
LABELS = ["human", "ai"]


class VoiceCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):  # x: (B, 1, n_mels, time)
        return self.head(self.features(x))


def make_melspec():
    import torchaudio

    return torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        ),
        torchaudio.transforms.AmplitudeToDB(top_db=80),
    )


def pick_device() -> torch.device:
    # ROCm PyTorch also reports as torch.cuda, so this works on AMD GPUs too.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
