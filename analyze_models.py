#!/usr/bin/env python3
"""
Model Analysis Utility for Josh Talks Speech & Audio Project
Analyze and compare saved model pickle files.
"""

import sys
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from model_evaluation import ModelManager

def analyze_models():
    """Analyze all saved models and generate reports."""
    print("📊 Josh Talks Model Analysis Report")
    print("=" * 60)
    
    model_manager = ModelManager()
    models = model_manager.list_saved_models()
    
    if not models:
        print("❌ No models found in models/ directory")
        print("Run 'python main_pipeline.py' or 'python model_demo.py' first")
        return
    
    print(f"Found {len(models)} saved models")
    print()
    
    # Create analysis DataFrame
    analysis_data = []
    
    for model in models:
        try:
            # Load detailed model data
            model_data = model_manager.load_model(model['filename'])
            metadata = model_data.get('training_metadata', {})
            
            analysis_data.append({
                'Model Name': model['custom_name'],
                'WER': metadata.get('validation_wer', 0),
                'Epochs': metadata.get('epochs', 0),
                'Learning Rate': metadata.get('learning_rate', 0),
                'Batch Size': metadata.get('batch_size', 0),
                'Dataset Size': metadata.get('dataset_size', 0),
                'Duration (hours)': metadata.get('total_duration_hours', 0),
                'Model Type': metadata.get('model_type', 'unknown'),
                'Optimization': metadata.get('optimization', 'none'),
                'File Size (MB)': model['file_size_mb'],
                'Created': model['created_timestamp']
            })
        except Exception as e:
            print(f"Warning: Could not analyze {model['filename']}: {e}")
    
    if not analysis_data:
        print("❌ Could not analyze any models")
        return
    
    df = pd.DataFrame(analysis_data)
    
    # Performance Analysis
    print("🎯 Performance Analysis:")
    print("-" * 40)
    
    best_model = df.loc[df['WER'].idxmin()]
    worst_model = df.loc[df['WER'].idxmax()]
    
    print(f"🥇 Best Model: {best_model['Model Name']}")
    print(f"   WER: {best_model['WER']:.3f}")
    print(f"   Epochs: {best_model['Epochs']}")
    print(f"   Dataset: {best_model['Dataset Size']} samples")
    print()
    
    print(f"🔍 Model Comparison:")
    print(f"   Best WER: {df['WER'].min():.3f}")
    print(f"   Worst WER: {df['WER'].max():.3f}")
    print(f"   Average WER: {df['WER'].mean():.3f}")
    print(f"   WER Range: {df['WER'].max() - df['WER'].min():.3f}")
    print()
    
    # Training Efficiency Analysis
    print("⚡ Training Efficiency:")
    print("-" * 40)
    
    # Models with training data
    trained_models = df[df['Epochs'] > 0]
    
    if len(trained_models) > 0:
        trained_models = trained_models.copy()
        baseline_wer = 0.830  # Pretrained model WER
        
        # Calculate improvement per epoch
        trained_models['WER_Improvement'] = baseline_wer - trained_models['WER']
        trained_models['Improvement_Per_Epoch'] = trained_models['WER_Improvement'] / trained_models['Epochs']
        trained_models['Relative_Improvement'] = (trained_models['WER_Improvement'] / baseline_wer) * 100
        
        most_efficient = trained_models.loc[trained_models['Improvement_Per_Epoch'].idxmax()]
        
        print(f"🚀 Most Efficient: {most_efficient['Model Name']}")
        print(f"   Improvement per epoch: {most_efficient['Improvement_Per_Epoch']:.4f}")
        print(f"   Total improvement: {most_efficient['Relative_Improvement']:.1f}%")
        print()
        
        # Resource usage
        print("💾 Resource Usage:")
        print(f"   Total storage: {df['File Size (MB)'].sum():.1f}MB")
        print(f"   Average file size: {df['File Size (MB)'].mean():.1f}MB")
        print(f"   Largest model: {df['File Size (MB)'].max():.1f}MB")
        print()
    
    # Detailed table
    print("📋 Detailed Model Comparison:")
    print("-" * 80)
    
    # Sort by WER performance
    df_sorted = df.sort_values('WER')
    
    print(f"{'Model':<25} {'WER':<8} {'Epochs':<7} {'Type':<12} {'Size(MB)':<10}")
    print("-" * 80)
    
    for _, row in df_sorted.iterrows():
        model_name = row['Model Name'][:24]
        print(f"{model_name:<25} {row['WER']:<8.3f} {row['Epochs']:<7} {row['Model Type']:<12} {row['File Size (MB)']:<10.1f}")
    
    # Save analysis to file
    report_path = "results/model_analysis_report.csv"
    df.to_csv(report_path, index=False)
    print(f"\n💾 Analysis saved to: {report_path}")
    
    # Training progression (if multiple versions exist)
    versioned_models = df[df['Model Name'].str.contains('_v')]
    if len(versioned_models) > 1:
        print("\n📈 Training Progression:")
        print("-" * 40)
        
        for _, row in versioned_models.sort_values('Created').iterrows():
            improvement = ((0.830 - row['WER']) / 0.830) * 100
            print(f"   {row['Model Name']}: {row['WER']:.3f} WER ({improvement:+.1f}%)")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 40)
    
    if len(trained_models) > 0:
        best_trained = trained_models.loc[trained_models['WER'].idxmin()]
        
        if best_trained['WER'] < 0.6:
            print("✅ Excellent performance achieved!")
            print("   Consider deploying the best model for production")
        elif best_trained['WER'] < 0.7:
            print("✅ Good performance achieved")
            print("   Consider additional training data or augmentation")
        else:
            print("⚠️  Performance needs improvement")
            print("   Recommendations:")
            print("   - Increase training data size")
            print("   - Try data augmentation techniques")
            print("   - Experiment with different learning rates")
    
    return df

def create_performance_visualization():
    """Create visualizations of model performance."""
    try:
        import matplotlib.pyplot as plt
        
        model_manager = ModelManager()
        models = model_manager.list_saved_models()
        
        if len(models) < 2:
            print("⚠️  Need at least 2 models for visualization")
            return
        
        # Extract data for plotting
        model_names = [m['custom_name'][:20] for m in models]  # Truncate names
        wers = [m['validation_wer'] for m in models if isinstance(m['validation_wer'], (int, float))]
        epochs = [m['epochs'] for m in models if isinstance(m['epochs'], (int, float))]
        
        if len(wers) != len(model_names):
            print("⚠️  Inconsistent data for visualization")
            return
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot 1: WER Comparison
        plt.subplot(2, 2, 1)
        plt.bar(range(len(model_names)), wers, color='skyblue', edgecolor='navy')
        plt.title('Model WER Comparison')
        plt.xlabel('Models')
        plt.ylabel('WER')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        
        # Add baseline line
        plt.axhline(y=0.83, color='red', linestyle='--', alpha=0.7, label='Baseline (0.83)')
        plt.legend()
        
        # Plot 2: Training Epochs vs WER
        plt.subplot(2, 2, 2)
        trained_models = [(e, w, n) for e, w, n in zip(epochs, wers, model_names) if e > 0]
        if trained_models:
            epochs_trained, wers_trained, names_trained = zip(*trained_models)
            plt.scatter(epochs_trained, wers_trained, color='green', s=100, alpha=0.7)
            
            # Add labels for points
            for i, name in enumerate(names_trained):
                plt.annotate(name, (epochs_trained[i], wers_trained[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.title('Training Epochs vs WER')
            plt.xlabel('Epochs')
            plt.ylabel('WER')
        
        # Plot 3: Improvement from baseline
        plt.subplot(2, 2, 3)
        baseline = 0.83
        improvements = [(baseline - w) / baseline * 100 for w in wers]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        plt.bar(range(len(model_names)), improvements, color=colors, alpha=0.7)
        plt.title('Improvement from Baseline (%)')
        plt.xlabel('Models')
        plt.ylabel('Improvement (%)')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 4: Model sizes
        plt.subplot(2, 2, 4)
        sizes = [m['file_size_mb'] for m in models]
        plt.bar(range(len(model_names)), sizes, color='orange', alpha=0.7)
        plt.title('Model File Sizes (MB)')
        plt.xlabel('Models')
        plt.ylabel('Size (MB)')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = "results/model_performance_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance visualization saved to: {plot_path}")
        
        # Don't show plot in automated context, just save it
        # plt.show()
        plt.close()
        
    except ImportError:
        print("⚠️  matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

if __name__ == "__main__":
    try:
        df = analyze_models()
        
        if df is not None and len(df) > 0:
            print("\n" + "="*60)
            create_performance_visualization()
            print("\n✅ Model analysis completed!")
            print("   Check results/ directory for saved reports and visualizations")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()