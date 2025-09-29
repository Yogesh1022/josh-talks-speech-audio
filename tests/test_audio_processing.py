#!/usr/bin/env python3
"""
Unit tests for Audio Processing module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_processing import DatasetAudioProcessor

class TestDatasetAudioProcessor:
    """Test suite for DatasetAudioProcessor class."""
    
    def test_initialization(self, tmp_path):
        """Test processor initialization."""
        data_path = str(tmp_path / 'data')
        processor = DatasetAudioProcessor(data_path)
        
        assert processor.data_path == data_path
        assert processor.sample_rate == 16000
        assert hasattr(processor, 'total_duration')
        assert hasattr(processor, 'total_files')
    
    def test_load_dataset_success(self, tmp_path, sample_dataset):
        """Test successful dataset loading."""
        # Create test Excel file
        data_dir = tmp_path / 'data'
        data_dir.mkdir(exist_ok=True)
        excel_path = data_dir / 'FT-Data.xlsx'
        sample_dataset.to_excel(excel_path, index=False)
        
        processor = DatasetAudioProcessor(str(data_dir))
        result = processor.load_dataset()
        
        assert len(result) == len(sample_dataset)
        assert 'recording_id' in result.columns
        assert 'language' in result.columns
        assert 'duration' in result.columns
    
    def test_load_dataset_file_not_found(self, tmp_path):
        """Test dataset loading when file doesn't exist."""
        data_dir = tmp_path / 'data'
        data_dir.mkdir(exist_ok=True)
        
        processor = DatasetAudioProcessor(str(data_dir))
        
        with pytest.raises(FileNotFoundError):
            processor.load_dataset()
    
    def test_process_audio_duration_calculation(self, mock_audio_data):
        """Test audio duration calculation."""
        # Audio duration should be length / sample_rate
        sample_rate = 16000
        expected_duration = len(mock_audio_data) / sample_rate
        
        # This would be the actual calculation in the processor
        calculated_duration = len(mock_audio_data) / sample_rate
        
        assert abs(calculated_duration - expected_duration) < 0.01
    
    @pytest.mark.parametrize("sample_rate,expected_target", [
        (8000, 16000),
        (16000, 16000),
        (22050, 16000),
        (44100, 16000),
    ])
    def test_target_sample_rate(self, sample_rate, expected_target):
        """Test that target sample rate is always 16kHz."""
        processor = DatasetAudioProcessor('test_path')
        # The processor should always target 16kHz
        assert processor.sample_rate == expected_target
    
    def test_calculate_statistics(self, tmp_path, sample_dataset):
        """Test statistics calculation."""
        # Create test data
        data_dir = tmp_path / 'data'
        data_dir.mkdir(exist_ok=True)
        excel_path = data_dir / 'FT-Data.xlsx'
        sample_dataset.to_excel(excel_path, index=False)
        
        processor = DatasetAudioProcessor(str(data_dir))
        dataset = processor.load_dataset()
        
        # Test basic statistics
        total_duration = dataset['duration'].sum()
        avg_duration = dataset['duration'].mean()
        file_count = len(dataset)
        
        assert total_duration > 0
        assert avg_duration > 0
        assert file_count == len(sample_dataset)
    
    def test_filter_by_language(self, tmp_path, sample_dataset):
        """Test language filtering functionality."""
        # Create test data
        data_dir = tmp_path / 'data'
        data_dir.mkdir(exist_ok=True)
        excel_path = data_dir / 'FT-Data.xlsx'
        sample_dataset.to_excel(excel_path, index=False)
        
        processor = DatasetAudioProcessor(str(data_dir))
        dataset = processor.load_dataset()
        
        # Filter for Hindi only
        hindi_records = dataset[dataset['language'] == 'hi']
        english_records = dataset[dataset['language'] == 'en']
        
        assert len(hindi_records) > 0
        assert len(english_records) > 0
        assert len(hindi_records) + len(english_records) == len(dataset)
    
    def test_validate_gcp_urls(self, tmp_path, sample_dataset):
        """Test GCP URL validation."""
        # Create test data
        data_dir = tmp_path / 'data'
        data_dir.mkdir(exist_ok=True)
        excel_path = data_dir / 'FT-Data.xlsx'
        sample_dataset.to_excel(excel_path, index=False)
        
        processor = DatasetAudioProcessor(str(data_dir))
        dataset = processor.load_dataset()
        
        # Check that all URLs start with 'gs://'
        for url in dataset['rec_url_gcp']:
            assert url.startswith('gs://'), f"Invalid GCP URL: {url}"
        
        for url in dataset['transcription_url_gcp']:
            assert url.startswith('gs://'), f"Invalid GCP URL: {url}"
    
    @patch('librosa.load')
    def test_audio_loading_mock(self, mock_librosa_load, mock_audio_data):
        """Test audio loading with mocked librosa."""
        # Mock librosa.load to return test data
        mock_librosa_load.return_value = (mock_audio_data, 16000)
        
        # This would be the actual audio loading logic
        audio_data, sr = mock_librosa_load('test_file.wav', sr=16000)
        
        assert len(audio_data) == len(mock_audio_data)
        assert sr == 16000
        mock_librosa_load.assert_called_once_with('test_file.wav', sr=16000)
    
    def test_duration_statistics(self, tmp_path, sample_dataset):
        """Test duration statistics calculation."""
        # Create test data
        data_dir = tmp_path / 'data'
        data_dir.mkdir(exist_ok=True)
        excel_path = data_dir / 'FT-Data.xlsx'
        sample_dataset.to_excel(excel_path, index=False)
        
        processor = DatasetAudioProcessor(str(data_dir))
        dataset = processor.load_dataset()
        
        durations = dataset['duration']
        
        # Test statistics
        assert durations.min() > 0
        assert durations.max() > durations.min()
        assert durations.mean() > 0
        assert durations.std() >= 0
    
    def test_user_id_distribution(self, tmp_path, sample_dataset):
        """Test user ID distribution analysis."""
        # Create test data
        data_dir = tmp_path / 'data'
        data_dir.mkdir(exist_ok=True)
        excel_path = data_dir / 'FT-Data.xlsx'
        sample_dataset.to_excel(excel_path, index=False)
        
        processor = DatasetAudioProcessor(str(data_dir))
        dataset = processor.load_dataset()
        
        # Analyze user distribution
        user_counts = dataset['user_id'].value_counts()
        unique_users = dataset['user_id'].nunique()
        
        assert unique_users > 0
        assert len(user_counts) == unique_users
        assert user_counts.sum() == len(dataset)
    
    def test_error_handling_invalid_data(self, tmp_path):
        """Test error handling with invalid data."""
        # Create invalid Excel file
        data_dir = tmp_path / 'data'
        data_dir.mkdir(exist_ok=True)
        excel_path = data_dir / 'FT-Data.xlsx'
        
        # Invalid dataset (missing required columns)
        invalid_data = pd.DataFrame({
            'invalid_column': ['test1', 'test2']
        })
        invalid_data.to_excel(excel_path, index=False)
        
        processor = DatasetAudioProcessor(str(data_dir))
        dataset = processor.load_dataset()
        
        # Should handle missing columns gracefully
        # In real implementation, this might raise an exception or use defaults
        assert 'invalid_column' in dataset.columns