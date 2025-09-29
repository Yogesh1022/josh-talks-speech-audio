#!/usr/bin/env python3
"""
Disfluency Detection Module for Josh Talks Speech & Audio Project
Detects and segments speech disfluencies in Hindi audio and transcriptions.
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import os
import json
from pathlib import Path
from audio_processing import WhisperAudioProcessor

class HindiDisfluencyDetector:
    """
    Detect various types of speech disfluencies in Hindi audio and text.

    Handles:
    - Filled Pauses (अं, उम्, etc.)
    - Repetitions (मैं-मैं, वो-वो, etc.)  
    - False Starts (जा—, कर—, etc.)
    - Prolongations (अच्छ्छ्छा, हम्म्म, etc.)
    - Self-Corrections (कल—, नहीं—, etc.)
    """

    def __init__(self, disfluency_patterns_file: str = "data/Speech-Disfluencies-List.xlsx"):
        """
        Initialize disfluency detector.

        Args:
            disfluency_patterns_file: Path to Excel file with disfluency patterns
        """
        self.patterns_file = disfluency_patterns_file
        self.patterns = self._load_disfluency_patterns()
        self.audio_processor = WhisperAudioProcessor()

    def _load_disfluency_patterns(self) -> Dict[str, List[str]]:
        """
        Load disfluency patterns from Excel file.

        Returns:
            Dictionary mapping disfluency types to pattern lists
        """
        try:
            df = pd.read_excel(self.patterns_file)
            patterns = {}

            for col in df.columns:
                # Get unique non-null patterns
                col_patterns = df[col].dropna().unique().tolist()
                # Clean and filter patterns
                clean_patterns = []
                for pattern in col_patterns:
                    if pd.notna(pattern) and str(pattern).strip():
                        clean_pattern = str(pattern).strip()
                        clean_patterns.append(clean_pattern)

                patterns[col] = clean_patterns

            return patterns

        except Exception as e:
            print(f"Error loading disfluency patterns: {e}")
            return {}

    def detect_text_disfluencies(self, text: str) -> List[Dict]:
        """
        Detect disfluencies in transcribed text.

        Args:
            text: Input transcription text

        Returns:
            List of detected disfluencies with metadata
        """
        detections = []

        # Normalize text
        normalized_text = self._normalize_hindi_text(text)

        for disfluency_type, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                # Find all occurrences of the pattern
                matches = self._find_pattern_matches(normalized_text, pattern)

                for match in matches:
                    detection = {
                        'type': disfluency_type,
                        'pattern': pattern,
                        'text': match['text'],
                        'start_char': match['start'],
                        'end_char': match['end'],
                        'confidence': self._calculate_confidence(pattern, match['text']),
                        'context': self._get_context(normalized_text, match['start'], match['end'])
                    }
                    detections.append(detection)

        # Sort by position in text
        detections.sort(key=lambda x: x['start_char'])

        return detections

    def _normalize_hindi_text(self, text: str) -> str:
        """
        Normalize Hindi text for pattern matching.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        return text

    def _find_pattern_matches(self, text: str, pattern: str) -> List[Dict]:
        """
        Find all matches of a pattern in text.

        Args:
            text: Input text
            pattern: Pattern to search for

        Returns:
            List of match dictionaries
        """
        matches = []

        # Escape pattern for regex safety, then find matches
        escaped_pattern = re.escape(pattern)

        for match in re.finditer(escaped_pattern, text, re.IGNORECASE):
            matches.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })

        return matches

    def _calculate_confidence(self, pattern: str, matched_text: str) -> float:
        """
        Calculate confidence score for disfluency detection.

        Args:
            pattern: Original pattern
            matched_text: Matched text

        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence based on exact match
        if pattern == matched_text:
            return 1.0
        elif pattern.lower() == matched_text.lower():
            return 0.9
        else:
            # Use string similarity
            similarity = len(set(pattern) & set(matched_text)) / len(set(pattern) | set(matched_text))
            return max(0.5, similarity)

    def _get_context(self, text: str, start: int, end: int, 
                    context_length: int = 50) -> str:
        """
        Get surrounding context for a detected disfluency.

        Args:
            text: Full text
            start: Start position of disfluency
            end: End position of disfluency
            context_length: Characters of context to include

        Returns:
            Context string
        """
        context_start = max(0, start - context_length)
        context_end = min(len(text), end + context_length)

        context = text[context_start:context_end]

        # Mark the disfluency in context
        relative_start = start - context_start
        relative_end = end - context_start

        marked_context = (
            context[:relative_start] + 
            "[" + context[relative_start:relative_end] + "]" +
            context[relative_end:]
        )

        return marked_context

    def estimate_audio_timestamps(self, text: str, audio_duration: float, 
                                detections: List[Dict]) -> List[Dict]:
        """
        Estimate audio timestamps for detected disfluencies.

        Args:
            text: Full transcription text
            audio_duration: Total audio duration in seconds
            detections: List of text-based detections

        Returns:
            Updated detections with estimated timestamps
        """
        if not detections or not text:
            return detections

        # Calculate characters per second
        chars_per_second = len(text) / audio_duration if audio_duration > 0 else 0

        for detection in detections:
            if chars_per_second > 0:
                # Estimate start and end times
                start_time = detection['start_char'] / chars_per_second
                end_time = detection['end_char'] / chars_per_second

                detection['start_time'] = round(start_time, 2)
                detection['end_time'] = round(end_time, 2)
                detection['duration'] = round(end_time - start_time, 2)
            else:
                detection['start_time'] = 0.0
                detection['end_time'] = 0.0
                detection['duration'] = 0.0

        return detections

    def extract_audio_segments(self, audio: np.ndarray, sr: int, 
                              detections: List[Dict], 
                              output_dir: str = "results/audio_clips") -> List[Dict]:
        """
        Extract audio segments for detected disfluencies.

        Args:
            audio: Audio array
            sr: Sample rate
            detections: List of detections with timestamps
            output_dir: Output directory for audio clips

        Returns:
            Updated detections with audio file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, detection in enumerate(detections):
            try:
                if 'start_time' in detection and 'end_time' in detection:
                    # Extract audio segment
                    segment = self.audio_processor.extract_segment(
                        audio, 
                        detection['start_time'], 
                        detection['end_time'], 
                        sr,
                        padding=0.5  # Add 0.5s padding
                    )

                    # Save segment
                    filename = f"disfluency_{i:03d}_{detection['type'].replace(' ', '_')}.wav"
                    output_path = os.path.join(output_dir, filename)

                    self.audio_processor.save_audio_segment(segment, output_path, sr)

                    # Update detection with file info
                    detection['audio_file'] = output_path
                    detection['audio_url'] = f"drive.google.com/{filename}"  # Simulated Google Drive URL

            except Exception as e:
                print(f"Error extracting audio segment {i}: {e}")
                detection['audio_file'] = None
                detection['audio_url'] = "extraction_failed"

        return detections

    def process_dataset(self, dataset_file: str = "data/FT-Data.xlsx") -> pd.DataFrame:
        """
        Process entire dataset for disfluency detection.

        Args:
            dataset_file: Path to FT-Data.xlsx

        Returns:
            DataFrame with disfluency detection results
        """
        try:
            df = pd.read_excel(dataset_file)
            all_detections = []

            for idx, row in df.iterrows():
                print(f"Processing recording {row['recording_id']} ({idx+1}/{len(df)})")

                # Get transcription (simulated)
                transcription = self._get_simulated_transcription(row['recording_id'])

                # Detect disfluencies in text
                detections = self.detect_text_disfluencies(transcription)

                # Estimate timestamps
                detections = self.estimate_audio_timestamps(
                    transcription, row['duration'], detections
                )

                # Add metadata
                for detection in detections:
                    detection.update({
                        'recording_id': row['recording_id'],
                        'user_id': row['user_id'],
                        'language': row['language'],
                        'full_transcription': transcription,
                        'total_duration': row['duration']
                    })

                all_detections.extend(detections)

            # Create results DataFrame
            results_df = pd.DataFrame(all_detections)

            return results_df

        except Exception as e:
            print(f"Error processing dataset: {e}")
            return pd.DataFrame()

    def _get_simulated_transcription(self, recording_id: int) -> str:
        """
        Generate simulated transcription with disfluencies for testing.

        Args:
            recording_id: Recording identifier

        Returns:
            Simulated transcription text
        """
        # Base sentences
        base_sentences = [
            "नमस्ते मैं आपको बताना चाहता हूं",
            "यह बहुत महत्वपूर्ण जानकारी है",
            "आजकल टेक्नोलॉजी तेजी से बढ़ रही है",
            "हमें अपने लक्ष्यों पर फोकस करना चाहिए",
            "सफलता के लिए मेहनत जरूरी है"
        ]

        # Add some disfluencies randomly
        disfluency_examples = ["अं", "उम्", "मैं-मैं", "वो-वो", "अच्छ्छा", "हम्म्म"]

        # Select base sentence
        base = np.random.choice(base_sentences)

        # Randomly insert disfluencies
        words = base.split()
        result_words = []

        for word in words:
            # Sometimes add disfluency before word
            if np.random.random() < 0.1:  # 10% chance
                result_words.append(np.random.choice(disfluency_examples))
            result_words.append(word)

        return " ".join(result_words)

    def create_results_sheet(self, detections: List[Dict], 
                           output_file: str = "results/disfluency_results.xlsx") -> pd.DataFrame:
        """
        Create results sheet in the required format.

        Args:
            detections: List of detection dictionaries
            output_file: Output Excel file path

        Returns:
            Results DataFrame
        """
        results = []

        for i, detection in enumerate(detections):
            result = {
                'disfluency_type': detection.get('type', 'unknown'),
                'audio_segment_url': detection.get('audio_url', f'drive.google.com/clip_{i}.wav'),
                'start_time (s)': detection.get('start_time', 0.0),
                'end_time (s)': detection.get('end_time', 0.0),
                'transcription_snippet': detection.get('text', ''),
                'notes': f"Confidence: {detection.get('confidence', 0.0):.2f}, Pattern: {detection.get('pattern', '')}"
            }
            results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save to Excel
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_excel(output_file, index=False)

        print(f"Results saved to {output_file}")
        print(f"Total detections: {len(results)}")

        # Print summary by type
        type_counts = df['disfluency_type'].value_counts()
        print("\nDetections by type:")
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count}")

        return df

def main():
    """Test the disfluency detection functionality."""
    print("Testing Disfluency Detection Module...")

    # Initialize detector
    detector = HindiDisfluencyDetector()

    # Test with sample text
    sample_text = "नमस्ते अं मैं-मैं आपको बताना चाहता हूं कि यह अच्छ्छा है उम्"

    print(f"Sample text: {sample_text}")

    # Detect disfluencies
    detections = detector.detect_text_disfluencies(sample_text)

    print(f"\nDetected {len(detections)} disfluencies:")
    for detection in detections:
        print(f"  Type: {detection['type']}")
        print(f"  Text: {detection['text']}")
        print(f"  Pattern: {detection['pattern']}")
        print(f"  Confidence: {detection['confidence']:.2f}")
        print(f"  Context: {detection['context']}")
        print()

    print("Disfluency detection test completed!")

if __name__ == "__main__":
    main()
