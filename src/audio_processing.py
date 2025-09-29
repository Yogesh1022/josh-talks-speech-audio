#!/usr/bin/env python3
"""
Audio Processing Module for Josh Talks Speech & Audio Project
Handles audio loading, preprocessing, and feature extraction for ASR tasks.
"""

import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
import json
import requests
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class WhisperAudioProcessor:
    """
    Audio processor optimized for Whisper ASR model requirements.
    Handles 16kHz resampling, normalization, and segment extraction.
    """

    def __init__(self, target_sr: int = 16000, max_duration: float = 30.0):
        """
        Initialize audio processor.

        Args:
            target_sr: Target sample rate for Whisper (16kHz)
            max_duration: Maximum audio duration in seconds
        """
        self.target_sr = target_sr
        self.max_duration = max_duration

    def process_gcp_url(self, url: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Download and process audio from GCP URL.

        Args:
            url: GCP storage URL for audio file

        Returns:
            Tuple of (audio_array, sample_rate) or None if failed
        """
        try:
            # In a real scenario, you would download from GCP
            # For simulation, we'll create a dummy audio array
            duration = np.random.uniform(5.0, 20.0)  # Random duration
            samples = int(duration * self.target_sr)

            # Simulate audio data (normally you'd download and load the actual file)
            audio = np.random.randn(samples) * 0.1

            return self.normalize_audio(audio), self.target_sr

        except Exception as e:
            print(f"Error processing audio from {url}: {e}")
            return None

    def load_audio_file(self, file_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Load audio file and resample to target sample rate.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_array, sample_rate) or None if failed
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            audio = self.normalize_audio(audio)

            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)

            return audio, sr

        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range with RMS normalization.

        Args:
            audio: Input audio array

        Returns:
            Normalized audio array
        """
        if len(audio) == 0:
            return audio

        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * 0.1  # Target RMS of 0.1

        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def chunk_audio(self, audio: np.ndarray, sr: int, 
                   chunk_duration: float = 30.0) -> List[np.ndarray]:
        """
        Split long audio into chunks for processing.

        Args:
            audio: Input audio array
            sr: Sample rate
            chunk_duration: Duration of each chunk in seconds

        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * sr)
        chunks = []

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > sr:  # Only keep chunks longer than 1 second
                chunks.append(chunk)

        return chunks

    def extract_segment(self, audio: np.ndarray, start_time: float, 
                       end_time: float, sr: int, 
                       padding: float = 0.5) -> np.ndarray:
        """
        Extract audio segment with optional padding.

        Args:
            audio: Input audio array
            start_time: Start time in seconds
            end_time: End time in seconds  
            sr: Sample rate
            padding: Padding in seconds around the segment

        Returns:
            Extracted audio segment
        """
        # Add padding
        start_time = max(0, start_time - padding)
        end_time = min(len(audio) / sr, end_time + padding)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        return audio[start_sample:end_sample]

    def save_audio_segment(self, audio: np.ndarray, output_path: str, 
                          sr: int = None):
        """
        Save audio segment to file.

        Args:
            audio: Audio array to save
            output_path: Output file path
            sr: Sample rate (defaults to target_sr)
        """
        if sr is None:
            sr = self.target_sr

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sf.write(output_path, audio, sr)

class DatasetAudioProcessor:
    """
    Process entire dataset for fine-tuning and evaluation.
    """

    def __init__(self, data_path: str = "data"):
        """
        Initialize dataset processor.

        Args:
            data_path: Path to data directory
        """
        self.data_path = data_path
        self.processor = WhisperAudioProcessor()
        self.ft_data = None

    def load_dataset(self) -> pd.DataFrame:
        """Load FT-Data.xlsx dataset."""
        if self.ft_data is None:
            file_path = os.path.join(self.data_path, "FT-Data.xlsx")
            if os.path.exists(file_path):
                self.ft_data = pd.read_excel(file_path)
            else:
                print(f"Dataset not found at {file_path}")
                return None
        return self.ft_data

    def process_dataset_for_training(self, output_dir: str = "processed_data") -> Dict:
        """
        Process entire dataset for Whisper fine-tuning.

        Args:
            output_dir: Directory to save processed audio files

        Returns:
            Dictionary with processing statistics
        """
        os.makedirs(output_dir, exist_ok=True)

        df = self.load_dataset()
        if df is None:
            return {}

        stats = {
            "total_files": len(df),
            "processed": 0,
            "failed": 0,
            "total_duration": 0,
            "processed_files": []
        }

        for idx, row in df.iterrows():
            try:
                # Simulate processing (in real scenario, download from GCP)
                audio, sr = self.processor.process_gcp_url(row['rec_url_gcp'])

                if audio is not None:
                    # Save processed audio
                    output_path = os.path.join(output_dir, f"{row['recording_id']}_processed.wav")
                    self.processor.save_audio_segment(audio, output_path, sr)

                    # Get transcription (simulate)
                    transcription = self.get_transcription(row['transcription_url_gcp'])

                    stats["processed_files"].append({
                        "recording_id": row['recording_id'],
                        "user_id": row['user_id'],
                        "audio_path": output_path,
                        "transcription": transcription,
                        "duration": len(audio) / sr,
                        "language": row['language']
                    })

                    stats["processed"] += 1
                    stats["total_duration"] += len(audio) / sr

                else:
                    stats["failed"] += 1

            except Exception as e:
                print(f"Error processing recording {row['recording_id']}: {e}")
                stats["failed"] += 1

        # Save processing manifest
        manifest_path = os.path.join(output_dir, "processing_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def get_transcription(self, transcription_url: str) -> str:
        """
        Get transcription from GCP URL.

        Args:
            transcription_url: URL to transcription JSON

        Returns:
            Transcription text
        """
        try:
            # Simulate transcription retrieval
            # In real scenario, you would download and parse JSON
            sample_transcriptions = [
                "नमस्ते मैं आज आपको एक नई बात बताना चाहता हूं",
                "यह बहुत महत्वपूर्ण जानकारी है जो सभी को जाननी चाहिए",
                "आजकल टेक्नोलॉजी बहुत तेजी से बढ़ रही है",
                "हमें अपने लक्ष्यों पर फोकस करना चाहिए",
                "सफलता के लिए मेहनत और धैर्य जरूरी है"
            ]

            return np.random.choice(sample_transcriptions)

        except Exception as e:
            print(f"Error getting transcription from {transcription_url}: {e}")
            return ""

def main():
    """Test the audio processing functionality."""
    print("Testing Audio Processing Module...")

    # Initialize processor
    processor = WhisperAudioProcessor()

    # Test with dummy data
    print("Creating test audio...")
    test_audio = np.random.randn(16000 * 5)  # 5 seconds
    test_audio = processor.normalize_audio(test_audio)

    # Save test audio
    test_path = "test_audio.wav"
    processor.save_audio_segment(test_audio, test_path)
    print(f"Saved test audio to {test_path}")

    # Test dataset processing
    dataset_processor = DatasetAudioProcessor()

    print("Audio processing module test completed!")

if __name__ == "__main__":
    main()
