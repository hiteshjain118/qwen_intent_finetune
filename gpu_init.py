#!/usr/bin/env python3
"""
GPU Initialization Module for Qwen1.5 Intent Classification

This module provides functions to download:
1. Qwen 1.5B model from Hugging Face
2. SGD (Schema-Guided Dialogue) dataset from GitHub

Based on the APIs used in the notebook.
"""

import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Error: {e.stderr}")
        return False

def download_qwen_model(model_id, local_dir):
    """
    Download Qwen model from Hugging Face.
    
    Args:
        model_id (str): Hugging Face model ID (default: "Qwen/Qwen1.5-1.8B")
        local_dir (str): Local directory to save the model (default: "./qwen_model_1.5B")
    
    Returns:
        bool: True if download successful, False otherwise
    """
    print("\n" + "="*60)
    print("DOWNLOADING QWEN MODEL")
    print("="*60)
    

    # Check if model already exists
    if os.path.exists(local_dir):
        print(f"✅ Model already exists at {local_dir}")
        return True
    
    # Download using huggingface_hub
    print("Downloading Qwen model...")

    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Download the model
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to {local_dir}")
    
    return True

def download_sgd_dataset(dataset_dir):
    """
    Download SGD dataset from GitHub.
    
    Args:
        dataset_dir (str): Local directory name for the dataset (default: "dstc8-schema-guided-dialogue")
    
    Returns:
        bool: True if download successful, False otherwise
    """
    print("\n" + "="*60)
    print("DOWNLOADING SGD DATASET")
    print("="*60)
    
    # Check if dataset already exists
    if os.path.exists(dataset_dir):
        print(f"✅ Dataset already exists at {dataset_dir}")
        return True
    
    # GitHub repository URL
    repo_url = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git"
    
    # Clone the repository to the specified directory
    success = run_command(f"git clone {repo_url} {dataset_dir}", "Cloning SGD dataset from GitHub")
    
    if success:
        print(f"✅ Dataset downloaded to {dataset_dir}")
        print(f"📁 Dataset structure:")
        
        # Show the structure of the downloaded dataset
        if os.path.exists(dataset_dir):
            for root, dirs, files in os.walk(dataset_dir):
                level = root.replace(dataset_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
                break  # Only show top level
    
    return success

def install_requirements(requirements_file="requirements.txt"):
    """
    Install Python packages from requirements.txt file.
    
    Args:
        requirements_file (str): Path to requirements.txt file (default: "requirements.txt")
    
    Returns:
        bool: True if installation successful, False otherwise
    """
    print("\n" + "="*60)
    print("INSTALLING REQUIREMENTS")
    print("="*60)
    
    # Check if requirements file exists
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    print(f"📦 Installing packages from {requirements_file}...")
    
    # Install requirements using pip
    success = run_command(f"pip install -r {requirements_file}", "Installing Python requirements")
    
    if success:
        print("✅ Requirements installed successfully!")
    else:
        print("❌ Failed to install requirements")
    
    return success

def verify_downloads(model_dir, dataset_dir):
    """
    Verify that both model and dataset were downloaded successfully.
    
    Args:
        model_dir (str): Path to model directory
        dataset_dir (str): Path to dataset directory
    
    Returns:
        dict: Dictionary with verification results
    """
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    results = {"model": False, "dataset": False}
    
    # Check model
    if os.path.exists(model_dir):
        model_files = os.listdir(model_dir)
        print(f"✅ Model directory exists: {model_dir}")
        print(f"   Found {len(model_files)} files")
        if model_files:
            print(f"   Sample files: {model_files[:3]}")
        results["model"] = True
    else:
        print(f"❌ Model directory missing: {model_dir}")
    
    # Check dataset
    if os.path.exists(dataset_dir):
        train_dir = os.path.join(dataset_dir, "train")
        if os.path.exists(train_dir):
            train_files = [f for f in os.listdir(train_dir) if f.endswith('.json')]
            print(f"✅ Dataset directory exists: {dataset_dir}")
            print(f"   Found {len(train_files)} training files")
            if train_files:
                print(f"   Sample files: {train_files[:3]}")
            results["dataset"] = True
        else:
            print(f"❌ Training directory missing: {train_dir}")
    else:
        print(f"❌ Dataset directory missing: {dataset_dir}")
    
    return results

def initialize_environment(check_directory=True, model_id=None, model_local_dir=None, dataset_dir=None, install_reqs=True, requirements_file="requirements.txt"):
    """
    Initialize the environment by installing requirements and downloading model and dataset.
    
    Args:
        check_directory (bool): Whether to check if running from finetune directory
        model_id (str): Hugging Face model ID
        model_local_dir (str): Local directory to save the model
        dataset_dir (str): Local directory name for the dataset
        install_reqs (bool): Whether to install requirements (default: True)
        requirements_file (str): Path to requirements.txt file (default: "requirements.txt")
    
    Returns:
        dict: Dictionary with initialization results
    """
    print("🚀 GPU Initialization for Qwen1.5 Intent Classification")
    print("This will install requirements and download the Qwen 1.5B model and SGD dataset")
    
    # Check if we're in the right directory
    if check_directory and not os.path.exists("finetune/qwen_intent_finetune.py"):
        print("❌ Please run this from the finetune directory")
        return {"success": False, "error": "Wrong directory"}
    
    # Install requirements first
    reqs_success = True
    if install_reqs:
        reqs_success = install_requirements(requirements_file)
    
    # Download model
    model_success = download_qwen_model(model_id, model_local_dir)
    
    # Download dataset
    dataset_success = download_sgd_dataset(dataset_dir)
    
    # Verify downloads
    verification_results = verify_downloads(model_local_dir, dataset_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_success = reqs_success and model_success and dataset_success
    
    if all_success:
        print("🎉 All downloads completed successfully!")
        print("\nNext steps:")
        print("1. Run the finetuning script: python qwen_intent_finetune.py")
        print("2. Or run evaluation: python evaluate_accuracy.py")
        print("3. Or run inference: python inference.py")
    else:
        print("⚠️  Some operations failed. Please check the errors above.")
        if not reqs_success:
            print("   - Requirements installation failed")
        if not model_success:
            print("   - Model download failed")
        if not dataset_success:
            print("   - Dataset download failed")
    
    return {
        "success": all_success,
        "requirements_installed": reqs_success,
        "model_downloaded": model_success,
        "dataset_downloaded": dataset_success,
        "verification": verification_results
    }

if __name__ == "__main__":
    initialize_environment(
        check_directory=True, 
        model_id="Qwen/Qwen1.5-1.8B", 
        model_local_dir="./qwen_model_1.5B_3", 
        dataset_dir="dstc8-schema-guided-dialogue_3",
        install_reqs=True,
        requirements_file="finetune/requirements.txt"
    )