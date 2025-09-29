#!/usr/bin/env python3
"""
Test configuration and fixtures for Josh Talks Speech & Audio Pipeline
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path

@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary data directory for all tests."""
    temp_dir = tempfile.mkdtemp(prefix="josh_talks_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="session")
def sample_dataset():
    """Create a comprehensive sample dataset for testing."""
    return pd.DataFrame({
        'recording_id': ['test_001', 'test_002', 'test_003', 'test_004', 'test_005'],
        'language': ['hi', 'hi', 'en', 'hi', 'en'],
        'duration': [10.5, 15.2, 8.7, 12.3, 20.1],
        'user_id': ['user1', 'user2', 'user1', 'user3', 'user2'],
        'rec_url_gcp': [
            'gs://bucket/test_001.wav',
            'gs://bucket/test_002.wav', 
            'gs://bucket/test_003.wav',
            'gs://bucket/test_004.wav',
            'gs://bucket/test_005.wav'
        ],
        'transcription_url_gcp': [
            'gs://bucket/trans_001.txt',
            'gs://bucket/trans_002.txt',
            'gs://bucket/trans_003.txt',
            'gs://bucket/trans_004.txt',
            'gs://bucket/trans_005.txt'
        ]
    })

@pytest.fixture(scope="session")
def sample_disfluencies():
    """Create sample disfluency patterns for testing."""
    return pd.DataFrame({
        'Disfluency_Type': [
            'Filler', 'Repetition', 'False_Start', 'Prolongation', 
            'Silent_Pause', 'Interjection'
        ],
        'Pattern': [
            'um', 'the the', 'I was- I am', 'sooo', '[pause]', 'you know'
        ],
        'Category': [
            'hesitation', 'repetition', 'restart', 'prolongation', 
            'pause', 'interjection'
        ],
        'Language': ['en', 'en', 'en', 'en', 'en', 'en'],
        'Example': [
            'I was um going to the store',
            'The the weather is nice today',
            'I was- I am happy to be here',
            'This is sooo good',
            'Well [pause] I think so',
            'This is you know really great'
        ]
    })

@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing."""
    # Generate 10 seconds of random audio at 16kHz
    sample_rate = 16000
    duration = 10
    samples = sample_rate * duration
    return np.random.randn(samples).astype(np.float32)

@pytest.fixture
def sample_texts():
    """Sample texts for testing transcription and disfluency detection."""
    return [
        "Hello world, this is a test.",
        "Um, I think this is, uh, quite good.",
        "The the weather is nice today.",
        "I was- I am happy to be here.",
        "Well, you know, this is really great.",
        "This is sooo interesting, don't you think?",
        "Can you please, um, help me with this?",
        "I I need to go to the store.",
        "Let me think about this for a moment.",
        "This is a normal sentence without disfluencies."
    ]

@pytest.fixture
def model_metadata_sample():
    """Sample model training metadata."""
    return {
        'epochs': 3,
        'learning_rate': 1e-5,
        'batch_size': 8,
        'validation_wer': 0.654,
        'training_completed': True,
        'dataset_size': 100,
        'total_duration_hours': 2.5,
        'model_type': 'whisper-small',
        'optimization': 'standard_training'
    }

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Set up test environment for each test."""
    # Create necessary directories
    test_dirs = ['data', 'results', 'models', 'processed_data']
    for dir_name in test_dirs:
        (tmp_path / dir_name).mkdir(exist_ok=True)
    
    # Change to temporary directory for tests
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    yield tmp_path
    
    # Restore original directory
    os.chdir(original_cwd)

@pytest.fixture
def create_test_excel_files(tmp_path, sample_dataset, sample_disfluencies):
    """Create test Excel files."""
    # Create FT-Data.xlsx
    ft_data_path = tmp_path / 'data' / 'FT-Data.xlsx'
    sample_dataset.to_excel(ft_data_path, index=False)
    
    # Create disfluency data
    disfluency_path = tmp_path / 'data' / 'Speech-Disfluencies-Result.xlsx'
    sample_disfluencies.to_excel(disfluency_path, index=False)
    
    return {
        'ft_data': str(ft_data_path),
        'disfluencies': str(disfluency_path)
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests (everything else)
        if not any(marker.name in ['slow', 'integration', 'gpu'] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)