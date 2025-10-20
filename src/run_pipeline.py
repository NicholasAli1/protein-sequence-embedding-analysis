#!/usr/bin/env python3
"""
Complete pipeline runner for Sequence Embedding Analysis.
Executes all steps: embedding generation, regression modeling, and visualization.
"""

import os
import sys
import subprocess
import time


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_command(cmd, cwd=None):
    """Run a command and handle errors."""
    try:
        subprocess.run(cmd, cwd=cwd, check=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Command failed with exit code {e.returncode}")
        return False


def main():
    """Run complete analysis pipeline."""
    
    start_time = time.time()
    
    print("\n" + "#"*70)
    print("# SEQUENCE EMBEDDING ANALYSIS PIPELINE")
    print("# UniRep & ProtBERT for Antibody Developability")
    print("#"*70)
    
    # Check if example data exists
    if not os.path.exists('../example_sequences.csv'):
        print("\n❌ Error: example_sequences.csv not found!")
        print("Please ensure the data file exists in the parent directory.")
        sys.exit(1)
    
    # Step 1: Generate Embeddings
    print_section("STEP 1/3: Generating Embeddings")
    print("This will generate UniRep and ProtBERT embeddings from sequences...")
    print("⏳ Note: First run downloads models (~1.6 GB for ProtBERT)")
    
    if not run_command("python3 generate_embeddings.py"):
        print("\n❌ Embedding generation failed. Exiting.")
        sys.exit(1)
    
    print("\n✅ Embeddings generated successfully!")
    
    # Check if embeddings were created
    if not os.path.exists('../data/embeddings.npz'):
        print("\n❌ Error: Embeddings file not found!")
        sys.exit(1)
    
    # Step 2: Train Regression Models
    print_section("STEP 2/3: Training Regression Models")
    print("Building Random Forest models to predict developability metrics...")
    
    if not run_command("python3 regression_model.py"):
        print("\n❌ Regression modeling failed. Exiting.")
        sys.exit(1)
    
    print("\n✅ Regression models trained successfully!")
    
    # Step 3: Generate Visualizations
    print_section("STEP 3/3: Creating Visualizations")
    print("Generating PCA, t-SNE, and UMAP visualizations...")
    
    if not run_command("python3 visualize_embeddings.py"):
        print("\n❌ Visualization failed. Exiting.")
        sys.exit(1)
    
    print("\n✅ Visualizations created successfully!")
    
    # Summary
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print_section("PIPELINE COMPLETE! 🎉")
    print(f"Total execution time: {minutes}m {seconds}s\n")
    
    print("📁 Generated Files:")
    print("   • data/embeddings.npz           - Sequence embeddings")
    print("   • models/                       - Trained regression models")
    print("   • plots/                        - All visualization outputs")
    
    print("\n📊 Next Steps:")
    print("   1. Review plots/ directory for visualizations")
    print("   2. Check model performance in plots/*/performance_summary_*.png")
    print("   3. Open interactive HTML files in plots/visualizations/")
    print("   4. Use trained models in models/ for predictions on new sequences")
    
    print("\n" + "="*70)
    print("Happy analyzing! 🧬")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
