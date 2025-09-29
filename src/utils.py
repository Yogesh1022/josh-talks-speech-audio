import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import os
import json
from pathlib import Path

class AudioProcessor:
    """Utility class for audio processing operations."""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load_and_resample(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            return audio, sr
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio

    def extract_segment(self, audio: np.ndarray, start_time: float, 
                       end_time: float, sr: int) -> np.ndarray:
        """Extract audio segment based on timestamps."""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        return audio[start_sample:end_sample]

    def save_audio_clip(self, audio: np.ndarray, output_path: str, sr: int):
        """Save audio clip to file."""
        sf.write(output_path, audio, sr)

class TextProcessor:
    """Utility class for text processing operations."""

    @staticmethod
    def clean_hindi_text(text: str) -> str:
        """Clean Hindi text for ASR processing."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove punctuation but keep Devanagari characters
        text = re.sub(r'[^ऀ-ॿ\s]', '', text)

        # Normalize unicode
        text = unicodedata.normalize('NFC', text)

        return text.strip()

    @staticmethod
    def is_hindi_text(text: str) -> bool:
        """Check if text contains Hindi/Devanagari characters."""
        hindi_chars = sum(1 for char in text if 'ऀ' <= char <= 'ॿ')
        return hindi_chars > len(text) * 0.5

    @staticmethod
    def detect_disfluencies(text: str, patterns: Dict[str, List[str]]) -> List[Dict]:
        """Detect disfluencies in text using pattern matching."""
        detected = []

        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                # Simple string matching (can be enhanced with regex)
                if pattern in text:
                    matches = [(m.start(), m.end()) for m in re.finditer(re.escape(pattern), text)]
                    for start, end in matches:
                        detected.append({
                            'type': category,
                            'pattern': pattern,
                            'start_char': start,
                            'end_char': end,
                            'text': text[start:end]
                        })

        return detected

class DatasetProcessor:
    """Utility class for dataset processing operations."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.ft_data = None
        self.disfluency_patterns = None

    def load_ft_data(self) -> pd.DataFrame:
        """Load fine-tuning dataset."""
        if self.ft_data is None:
            self.ft_data = pd.read_excel(os.path.join(self.data_path, 'FT-Data.xlsx'))
        return self.ft_data

    def load_disfluency_patterns(self) -> Dict[str, List[str]]:
        """Load disfluency patterns from Excel."""
        if self.disfluency_patterns is None:
            df = pd.read_excel(os.path.join(self.data_path, 'Speech-Disfluencies-List.xlsx'))

            patterns = {}
            for col in df.columns:
                patterns[col] = df[col].dropna().unique().tolist()
                patterns[col] = [str(p).strip() for p in patterns[col] if str(p).strip()]

            self.disfluency_patterns = patterns

        return self.disfluency_patterns

    def create_train_val_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/validation split ensuring user diversity."""
        ft_data = self.load_ft_data()

        # Split by users to avoid data leakage
        unique_users = ft_data['user_id'].unique()
        np.random.shuffle(unique_users)

        n_val_users = int(len(unique_users) * test_size)
        val_users = unique_users[:n_val_users]
        train_users = unique_users[n_val_users:]

        train_data = ft_data[ft_data['user_id'].isin(train_users)]
        val_data = ft_data[ft_data['user_id'].isin(val_users)]

        return train_data, val_data

class EvaluationUtils:
    """Utility class for model evaluation."""

    @staticmethod
    def calculate_wer(predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate using jiwer."""
        try:
            from jiwer import wer
            return wer(references, predictions)
        except ImportError:
            print("jiwer not installed. Install with: pip install jiwer")
            return None

    @staticmethod
    def estimate_timestamps(text: str, duration: float) -> List[Tuple[float, float, str]]:
        """Estimate word-level timestamps assuming uniform speech rate."""
        words = text.split()
        if not words:
            return []

        time_per_word = duration / len(words)
        timestamps = []

        for i, word in enumerate(words):
            start_time = i * time_per_word
            end_time = (i + 1) * time_per_word
            timestamps.append((start_time, end_time, word))

        return timestamps

class ResultExporter:
    """Utility class for exporting results."""

    @staticmethod
    def export_to_excel(data: Dict, filename: str, sheet_name: str = 'Results'):
        """Export results to Excel file."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def create_disfluency_result_sheet(detections: List[Dict], output_path: str):
        """Create disfluency results in the required format."""
        results = []

        for detection in detections:
            results.append({
                'disfluency_type': detection['type'],
                'audio_segment_url': f"drive.google.com/clip_{detection.get('id', 'unknown')}.wav",
                'start_time (s)': detection.get('start_time', 0.0),
                'end_time (s)': detection.get('end_time', 0.0),
                'transcription_snippet': detection.get('text', ''),
                'notes': detection.get('notes', '')
            })

        df = pd.DataFrame(results)
        df.to_excel(output_path, index=False)

        return df

def setup_directories():
    """Create necessary project directories."""
    directories = [
        'data', 'notebooks', 'src', 'results', 'results/audio_clips', 'models'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    # Test utilities
    setup_directories()
    print("Project utilities initialized successfully!")
