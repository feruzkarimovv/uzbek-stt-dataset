# Uzbek Speech-to-Text Dataset

A small-scale Uzbek language speech-to-text dataset containing 74 audio clips with manual transcriptions.

## Dataset Overview

This dataset was created for training and evaluating Uzbek speech recognition models. It contains high-quality audio recordings with transcriptions in Uzbek language using Latin script.

### Statistics

- **Total clips:** 74
- **Train set:** 59 clips (80%)
- **Validation set:** 15 clips (20%)
- **Total duration:** Approximately 7-8 minutes
- **Audio format:** WAV, 16kHz, mono
- **Language:** Uzbek (Latin script)
- **Clip duration:** 3-10 seconds

## Dataset Structure

```
data/
├── train/
│   ├── clip_002.wav
│   ├── clip_003.wav
│   └── ... (59 files)
├── val/
│   ├── clip_005.wav
│   ├── clip_006.wav
│   └── ... (14 files)
├── train_metadata.csv
└── val_metadata.csv
```

## Metadata Format

Each metadata CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| file_name | Audio file name |
| text | Uzbek transcription in Latin script |
| duration_seconds | Audio duration in seconds |
| source_file | Original source file name |

### Example

```csv
file_name,text,duration_seconds,source_file
clip_002.wav,"Kattalar hayotidagi qo'pol va manfaatli olamning to'qnashuvi haqida",3.48,"source.wav"
```

## Usage

### Python Example

```python
import pandas as pd

# Load metadata
train_df = pd.read_csv('data/train_metadata.csv')
val_df = pd.read_csv('data/val_metadata.csv')

# Iterate through dataset
for idx, row in train_df.iterrows():
    audio_path = f"data/train/{row['file_name']}"
    text = row['text']
    print(f"{audio_path}: {text}")
```

### With PyTorch

```python
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

class UzbekSTTDataset(Dataset):
    def __init__(self, metadata_path, audio_dir):
        self.data = pd.read_csv(metadata_path)
        self.audio_dir = audio_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = f"{self.audio_dir}/{row['file_name']}"
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, row['text']

# Load dataset
train_dataset = UzbekSTTDataset('data/train_metadata.csv', 'data/train')
val_dataset = UzbekSTTDataset('data/val_metadata.csv', 'data/val')
```

## Creation Process

1. **Collection:** Audio collected from Uzbek language sources
2. **Segmentation:** Automatic speech activity detection and segmentation
3. **Transcription:** Initial automatic transcription using Whisper
4. **Manual Review:** All transcriptions manually reviewed and corrected
5. **Quality Control:** Removed low-quality or unclear audio clips
6. **Split:** Random 80/20 split into training and validation sets

## Quality Standards

- Clear speech with minimal background noise
- Accurate Uzbek transcriptions
- Proper grammar and spelling
- Consistent audio quality across all clips
- Balanced duration distribution
