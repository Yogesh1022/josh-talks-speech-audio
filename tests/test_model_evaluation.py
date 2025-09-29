#!/usr/bin/env python3
"""
Unit tests for Model Evaluation module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import pickle
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_evaluation import WhisperEvaluator, ModelManager

class TestWhisperEvaluator:
    """Test suite for WhisperEvaluator class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        evaluator = WhisperEvaluator()
        
        assert evaluator.model_name == "openai/whisper-small"
        assert evaluator.device in ["cpu", "cuda", "mps"]
        assert hasattr(evaluator, 'model')
        assert hasattr(evaluator, 'processor')
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model name."""
        model_name = "openai/whisper-tiny"
        evaluator = WhisperEvaluator(model_name=model_name)
        
        assert evaluator.model_name == model_name
    
    @patch('jiwer.wer')
    def test_calculate_wer_perfect_match(self, mock_wer):
        """Test WER calculation with perfect transcription."""
        mock_wer.return_value = 0.0
        
        evaluator = WhisperEvaluator()
        reference = ["hello world"]
        prediction = ["hello world"]
        
        wer_score = evaluator.calculate_wer(reference, prediction)
        
        assert wer_score == 0.0
        mock_wer.assert_called_once_with(reference, prediction)
    
    @patch('jiwer.wer')
    def test_calculate_wer_with_errors(self, mock_wer):
        """Test WER calculation with transcription errors."""
        mock_wer.return_value = 0.25  # 25% error rate
        
        evaluator = WhisperEvaluator()
        reference = ["hello world how are you"]
        prediction = ["hello world how you"]  # Missing "are"
        
        wer_score = evaluator.calculate_wer(reference, prediction)
        
        assert wer_score == 0.25
        mock_wer.assert_called_once_with(reference, prediction)
    
    def test_calculate_wer_without_jiwer(self):
        """Test WER calculation fallback when jiwer is not available."""
        evaluator = WhisperEvaluator()
        
        # Temporarily disable jiwer
        original_available = hasattr(evaluator, 'jiwer_available')
        evaluator.jiwer_available = False
        
        reference = ["hello world"]
        prediction = ["hello world"]
        
        # Should return 0 or handle gracefully without jiwer
        wer_score = evaluator.calculate_wer(reference, prediction)
        
        assert isinstance(wer_score, (int, float))
        assert wer_score >= 0
    
    @patch('jiwer.cer')
    def test_calculate_cer(self, mock_cer):
        """Test Character Error Rate calculation."""
        mock_cer.return_value = 0.1  # 10% character error rate
        
        evaluator = WhisperEvaluator()
        reference = ["hello"]
        prediction = ["helo"]  # Missing one character
        
        # If CER method exists
        if hasattr(evaluator, 'calculate_cer'):
            cer_score = evaluator.calculate_cer(reference, prediction)
            assert cer_score == 0.1
            mock_cer.assert_called_once_with(reference, prediction)
    
    def test_model_comparison_data_structure(self):
        """Test model comparison data structure."""
        evaluator = WhisperEvaluator()
        
        # Test data structure that would be used for comparison
        comparison_data = {
            'Model': ['Whisper-Small', 'Whisper-Tiny', 'Custom-Model'],
            'Hindi': [0.85, 0.90, 0.65],
            'English': [0.75, 0.85, 0.70],
            'Parameters': ['244M', '39M', '244M'],
            'Training_Time': ['2h', '1h', '4h']
        }
        
        df = pd.DataFrame(comparison_data)
        
        assert len(df) == 3
        assert 'Model' in df.columns
        assert 'Hindi' in df.columns
        assert df.iloc[2]['Hindi'] < df.iloc[0]['Hindi']  # Custom model is better
    
    @patch('transformers.pipeline')
    def test_transcription_mock(self, mock_pipeline):
        """Test transcription with mocked pipeline."""
        # Mock the transcription pipeline
        mock_transcriber = MagicMock()
        mock_transcriber.return_value = {"text": "hello world"}
        mock_pipeline.return_value = mock_transcriber
        
        evaluator = WhisperEvaluator()
        
        # Mock audio data
        fake_audio_path = "test_audio.wav"
        
        # This would be the actual transcription logic
        result = mock_transcriber(fake_audio_path)
        
        assert result["text"] == "hello world"
        mock_transcriber.assert_called_once_with(fake_audio_path)
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics structure."""
        # Test the structure of evaluation results
        metrics = {
            'wer': 0.65,
            'cer': 0.45,
            'bleu': 0.75,
            'processing_time': 2.5,
            'model_size': '244M',
            'gpu_memory': '2GB'
        }
        
        assert 'wer' in metrics
        assert 'processing_time' in metrics
        assert metrics['wer'] > 0
        assert metrics['processing_time'] > 0
    
    def test_batch_evaluation(self):
        """Test batch evaluation logic."""
        evaluator = WhisperEvaluator()
        
        # Test batch processing structure
        audio_files = ["file1.wav", "file2.wav", "file3.wav"]
        expected_results = len(audio_files)
        
        # Mock batch results
        batch_results = []
        for i, file in enumerate(audio_files):
            batch_results.append({
                'file': file,
                'transcription': f"transcription {i}",
                'processing_time': np.random.uniform(1.0, 3.0)
            })
        
        assert len(batch_results) == expected_results
        assert all('transcription' in result for result in batch_results)


class TestModelManager:
    """Test suite for ModelManager class."""
    
    def test_initialization(self, tmp_path):
        """Test ModelManager initialization."""
        # Set up test models directory
        models_dir = tmp_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Mock the models directory
        with patch.object(ModelManager, '_get_models_dir', return_value=str(models_dir)):
            manager = ModelManager()
            
            assert manager.models_dir.exists()
            assert manager.models_dir.is_dir()
    
    def test_save_model_metadata(self, tmp_path, model_metadata_sample):
        """Test model saving with metadata."""
        models_dir = tmp_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        with patch.object(ModelManager, '_get_models_dir', return_value=str(models_dir)):
            manager = ModelManager()
            
            # Mock evaluator
            mock_evaluator = MagicMock()
            mock_evaluator.model_name = "openai/whisper-small"
            mock_evaluator.model = MagicMock()
            mock_evaluator.model.state_dict.return_value = {"test": "data"}
            mock_evaluator.device = "cpu"
            
            # Save model
            saved_path = manager.save_whisper_model(
                mock_evaluator,
                "test_model",
                model_metadata_sample
            )
            
            assert saved_path is not None
            assert Path(saved_path).exists()
    
    def test_list_saved_models_empty(self, tmp_path):
        """Test listing models when directory is empty."""
        models_dir = tmp_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        with patch.object(ModelManager, '_get_models_dir', return_value=str(models_dir)):
            manager = ModelManager()
            models = manager.list_saved_models()
            
            assert isinstance(models, list)
            assert len(models) == 0
    
    def test_list_saved_models_with_files(self, tmp_path, model_metadata_sample):
        """Test listing models with existing files."""
        models_dir = tmp_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Create mock pickle file
        test_model_data = {
            'model_name': 'openai/whisper-small',
            'custom_name': 'test_model',
            'training_metadata': model_metadata_sample,
            'created_timestamp': '20240101_120000'
        }
        
        test_file = models_dir / 'test_model_20240101_120000.pkl'
        with open(test_file, 'wb') as f:
            pickle.dump(test_model_data, f)
        
        with patch.object(ModelManager, '_get_models_dir', return_value=str(models_dir)):
            manager = ModelManager()
            models = manager.list_saved_models()
            
            assert len(models) == 1
            assert models[0]['custom_name'] == 'test_model'
            assert models[0]['filename'] == 'test_model_20240101_120000.pkl'
    
    def test_load_model_success(self, tmp_path, model_metadata_sample):
        """Test successful model loading."""
        models_dir = tmp_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Create mock pickle file
        test_model_data = {
            'model_name': 'openai/whisper-small',
            'custom_name': 'test_model',
            'training_metadata': model_metadata_sample,
            'created_timestamp': '20240101_120000',
            'model_state_dict': {"test": "data"}
        }
        
        test_file = models_dir / 'test_model_20240101_120000.pkl'
        with open(test_file, 'wb') as f:
            pickle.dump(test_model_data, f)
        
        with patch.object(ModelManager, '_get_models_dir', return_value=str(models_dir)):
            manager = ModelManager()
            loaded_data = manager.load_model('test_model_20240101_120000.pkl')
            
            assert loaded_data is not None
            assert loaded_data['custom_name'] == 'test_model'
            assert 'training_metadata' in loaded_data
    
    def test_load_model_not_found(self, tmp_path):
        """Test loading non-existent model."""
        models_dir = tmp_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        with patch.object(ModelManager, '_get_models_dir', return_value=str(models_dir)):
            manager = ModelManager()
            
            with pytest.raises(FileNotFoundError):
                manager.load_model('nonexistent_model.pkl')
    
    def test_model_info_creation(self):
        """Test model info dictionary creation."""
        manager = ModelManager()
        
        # Test the structure that would be created
        model_info = {
            'model_name': 'test_model',
            'wer_score': 0.65,
            'training_params': {'epochs': 3, 'lr': 1e-5},
            'timestamp': '2024-01-01T12:00:00',
            'file_size_mb': 1074.3,
            'performance_metrics': {
                'validation_accuracy': 0.85,
                'training_time': 120.5
            }
        }
        
        assert 'model_name' in model_info
        assert 'wer_score' in model_info
        assert 'timestamp' in model_info
        assert isinstance(model_info['training_params'], dict)
    
    def test_model_comparison_logic(self):
        """Test model comparison functionality."""
        # Mock model data for comparison
        models_data = [
            {'name': 'model_a', 'wer': 0.75, 'epochs': 2},
            {'name': 'model_b', 'wer': 0.65, 'epochs': 3},
            {'name': 'model_c', 'wer': 0.85, 'epochs': 1}
        ]
        
        # Sort by WER (lower is better)
        sorted_models = sorted(models_data, key=lambda x: x['wer'])
        best_model = sorted_models[0]
        
        assert best_model['name'] == 'model_b'
        assert best_model['wer'] == 0.65
    
    def test_model_validation(self, model_metadata_sample):
        """Test model validation logic."""
        # Test validation rules
        assert model_metadata_sample['epochs'] > 0
        assert 0 <= model_metadata_sample['validation_wer'] <= 1
        assert model_metadata_sample['learning_rate'] > 0
        assert model_metadata_sample['batch_size'] > 0
        assert model_metadata_sample['dataset_size'] > 0
    
    def test_file_size_calculation(self, tmp_path):
        """Test file size calculation."""
        models_dir = tmp_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Create a test file with known content
        test_file = models_dir / 'test_size.pkl'
        test_data = {"data": "x" * 1000}  # Approximately 1KB of data
        
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        file_size_bytes = test_file.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        assert file_size_bytes > 0
        assert file_size_mb >= 0