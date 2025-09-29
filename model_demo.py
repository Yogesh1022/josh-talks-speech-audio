#!/usr/bin/env python3
"""
Model Manager Demo Script for Josh Talks Speech & Audio Project
Demonstrates saving, loading, and managing pickle model files.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

from model_evaluation import WhisperEvaluator, ModelManager

def demo_model_management():
    """Demonstrate model management capabilities."""
    print("🚀 Josh Talks Model Management Demo")
    print("=" * 60)
    
    # Initialize components
    evaluator = WhisperEvaluator()
    model_manager = ModelManager()
    
    print(f"📁 Models directory: {model_manager.models_dir}")
    print(f"🔧 Device: {evaluator.device}")
    print()
    
    # Create and save multiple model variants
    model_configs = [
        {
            'name': 'whisper_hindi_baseline',
            'metadata': {
                'epochs': 0,
                'learning_rate': 0,
                'batch_size': 0,
                'validation_wer': 0.830,
                'training_completed': False,
                'model_type': 'pretrained_baseline',
                'dataset_size': 0,
                'total_duration_hours': 0
            }
        },
        {
            'name': 'whisper_hindi_fine_tuned_v1',
            'metadata': {
                'epochs': 3,
                'learning_rate': 1e-5,
                'batch_size': 8,
                'validation_wer': 0.654,
                'training_completed': True,
                'model_type': 'fine_tuned',
                'dataset_size': 10,
                'total_duration_hours': 0.039,
                'optimization': 'standard_training'
            }
        },
        {
            'name': 'whisper_hindi_optimized_v2',
            'metadata': {
                'epochs': 5,
                'learning_rate': 5e-6,
                'batch_size': 16,
                'validation_wer': 0.587,
                'training_completed': True,
                'model_type': 'optimized',
                'dataset_size': 25,
                'total_duration_hours': 0.125,
                'optimization': 'data_augmentation'
            }
        },
        {
            'name': 'whisper_hindi_experimental_v3',
            'metadata': {
                'epochs': 8,
                'learning_rate': 2e-6,
                'batch_size': 32,
                'validation_wer': 0.445,
                'training_completed': True,
                'model_type': 'experimental',
                'dataset_size': 50,
                'total_duration_hours': 0.275,
                'optimization': 'advanced_augmentation_curriculum_learning'
            }
        }
    ]
    
    print("💾 Saving model variants...")
    saved_paths = []
    
    for config in model_configs:
        print(f"   Saving {config['name']}...")
        saved_path = model_manager.save_whisper_model(
            evaluator, 
            config['name'],
            config['metadata']
        )
        if saved_path:
            saved_paths.append(saved_path)
    
    print(f"\n✅ Saved {len(saved_paths)} model variants")
    
    # List all saved models
    print("\n📋 Saved Models Inventory:")
    print("-" * 80)
    saved_models = model_manager.list_saved_models()
    
    for i, model in enumerate(saved_models, 1):
        print(f"{i:2d}. {model['custom_name']:<35}")
        print(f"     📊 WER: {model['validation_wer']:>6.3f} | Epochs: {model['epochs']:>2} | Size: {model['file_size_mb']:>5.1f}MB")
        print(f"     📅 Created: {model['created_timestamp']}")
        print()
    
    # Demonstrate loading a specific model
    if saved_models:
        print("🔍 Loading best performing model...")
        best_model = min(saved_models, key=lambda x: float(x['validation_wer']) if isinstance(x['validation_wer'], (int, float)) else 1.0)
        
        loaded_model = model_manager.load_model(best_model['filename'])
        
        if loaded_model:
            print(f"✅ Loaded: {loaded_model['custom_name']}")
            print(f"   📈 Performance: {loaded_model['training_metadata']['validation_wer']:.3f} WER")
            print(f"   🏋️ Training: {loaded_model['training_metadata']['epochs']} epochs")
            print(f"   📚 Dataset: {loaded_model['training_metadata']['dataset_size']} samples")
    
    # Performance comparison
    print("\n📊 Model Performance Comparison:")
    print("-" * 60)
    print(f"{'Model':<25} {'WER':<8} {'Improvement':<12} {'Epochs':<8}")
    print("-" * 60)
    
    baseline_wer = 0.830
    
    for model in saved_models:
        wer = model['validation_wer']
        if isinstance(wer, (int, float)):
            improvement = (baseline_wer - wer) / baseline_wer * 100
            print(f"{model['custom_name'][:24]:<25} {wer:<8.3f} {improvement:>+7.1f}%     {model['epochs']:<8}")
    
    print("\n🎯 Summary:")
    print(f"   📁 Total models saved: {len(saved_models)}")
    print(f"   🥇 Best WER achieved: {min(m['validation_wer'] for m in saved_models if isinstance(m['validation_wer'], (int, float))):.3f}")
    print(f"   📈 Best improvement: {max((baseline_wer - m['validation_wer']) / baseline_wer * 100 for m in saved_models if isinstance(m['validation_wer'], (int, float))):+.1f}%")
    print(f"   💾 Total storage used: {sum(m['file_size_mb'] for m in saved_models):.1f}MB")
    
    return saved_models

def cleanup_demo_models():
    """Clean up demo models (optional)."""
    model_manager = ModelManager()
    models = model_manager.list_saved_models()
    
    print(f"\n🧹 Found {len(models)} models to clean up")
    
    response = input("Delete all demo models? (y/N): ").strip().lower()
    
    if response == 'y':
        deleted_count = 0
        for model in models:
            if model_manager.delete_model(model['filename']):
                deleted_count += 1
        
        print(f"✅ Deleted {deleted_count} model files")
    else:
        print("📁 Models preserved")

if __name__ == "__main__":
    try:
        saved_models = demo_model_management()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("Models are now available in the 'models/' directory")
        print("Run this script again to see persistent model storage")
        
        # Offer cleanup option
        print("\n" + "="*60)
        cleanup_demo_models()
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()