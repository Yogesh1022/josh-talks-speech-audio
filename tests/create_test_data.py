#!/usr/bin/env python3
"""
Create test data for CI/CD pipeline
"""

import pandas as pd
import numpy as np
import os

def create_test_data():
    """Create test data files for pipeline testing."""
    
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("📁 Creating test data directories...")
    
    # Create FT-Data.xlsx
    ft_data = pd.DataFrame({
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
    
    ft_data.to_excel('data/FT-Data.xlsx', index=False)
    print("✅ Created data/FT-Data.xlsx")
    
    # Create disfluency data
    disfluency_data = pd.DataFrame({
        'Disfluency_Type': [
            'Filler', 'Repetition', 'False_Start', 'Prolongation'
        ],
        'Pattern': [
            'um', 'the the', 'I was- I am', 'sooo'
        ],
        'Category': [
            'hesitation', 'repetition', 'restart', 'prolongation'
        ],
        'Language': ['en', 'en', 'en', 'en'],
        'Example': [
            'I was um going to the store',
            'The the weather is nice today',
            'I was- I am happy to be here',
            'This is sooo good'
        ]
    })
    
    disfluency_data.to_excel('data/Speech-Disfluencies-Result.xlsx', index=False)
    print("✅ Created data/Speech-Disfluencies-Result.xlsx")
    
    print("🎯 Test data creation completed successfully!")
    return True

if __name__ == "__main__":
    try:
        create_test_data()
    except Exception as e:
        print(f"❌ Error creating test data: {e}")
        exit(1)