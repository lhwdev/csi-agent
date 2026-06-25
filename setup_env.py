#!/usr/bin/env python3
# ==============================================================================
# LeRobot Environment Setup & Initialization Script
# ==============================================================================
# This script automates the installation of LeRobot dependencies, Git LFS,
# and Jupyter widget extensions inside an active Conda environment.
#
# Usage:
#   python lhwdev/setup_env.py
# ==============================================================================

import sys
import os
import subprocess
import shutil
from pathlib import Path

# Color formatting for beautiful terminal output
def print_step(msg):
    print(f"\n\033[1;36m==> {msg}\033[0m")

def print_success(msg):
    print(f"\033[1;32m✓ {msg}\033[0m")

def print_warning(msg):
    print(f"\033[1;33m⚠️ {msg}\033[0m")

def print_error(msg):
    print(f"\033[1;31m✗ {msg}\033[0m")

def run_command(cmd, shell=True, check=True):
    """Helper to run shell commands and display output in real-time."""
    print(f"\033[90mRunning: {cmd}\033[0m")
    result = subprocess.run(cmd, shell=shell, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result

def main():
    print("\n\033[1;35m====================================================")
    print(" 🚀 LeRobot Conda Environment Initializer ")
    print("====================================================\033[0m")

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    print(f"Working directory set to repository root: {repo_root}")

    # Step 1: Check Conda Environment
    print_step("Step 1: Checking Conda Environment")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_env or not conda_prefix:
        print_warning("You do not appear to be running inside an active Conda environment!")
        print("Please create and activate your environment first, for example:")
        print("  conda create -n lerobot python=3.12 -y")
        print("  conda activate lerobot")
        choice = input("\nDo you want to proceed anyway? (y/n): ").strip().lower()
        if choice != 'y':
            print_error("Setup aborted by user.")
            sys.exit(1)
    else:
        print_success(f"Active Conda environment detected: '{conda_env}' (Path: {conda_prefix})")

    # Step 2: Detect or Install Package Manager (uv / pip)
    print_step("Step 2: Detecting Package Manager")
    has_uv = shutil.which("uv") is not None
    
    if not has_uv:
        print_warning("'uv' (ultra-fast package installer) was not found in your PATH.")
        choice = input("Would you like to install 'uv' inside this environment for 10x faster setup? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                print("Installing 'uv' via pip...")
                run_command(f"{sys.executable} -m pip install uv")
                has_uv = True
                print_success("'uv' successfully installed!")
            except Exception as e:
                print_warning(f"Failed to install 'uv' ({e}). Falling back to standard 'pip'.")
        else:
            print("Proceeding with standard 'pip'.")

    # Step 3: Install LeRobot & Notebook dependencies
    print_step("Step 3: Installing Dependencies")
    try:
        if has_uv:
            print("Using 'uv' to sync and install dependencies...")
            # Sync locked dependencies and install all extras (including dev, test, and optional hardware extras)
            run_command("uv sync --locked --extra all")
            # Ensure ipywidgets is present in the environment for the GUI
            run_command("uv pip install ipywidgets opencv-python")
        else:
            print("Using standard 'pip' to install dependencies...")
            # Install the package in editable mode with all extras
            run_command(f"{sys.executable} -m pip install -e .[all,dev,test]")
            # Ensure ipywidgets is present
            run_command(f"{sys.executable} -m pip install ipywidgets opencv-python")
        print_success("Dependencies successfully installed!")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed during package installation:\nStdout: {e.stdout}\nStderr: {e.stderr}")
        sys.exit(1)

    # Step 4: Setup Git LFS for large assets/models
    print_step("Step 4: Setting up Git LFS (Large File Storage)")
    if shutil.which("git") is not None:
        try:
            # Check if git lfs is installed on the system
            lfs_check = subprocess.run("git lfs version", shell=True, capture_output=True)
            if lfs_check.returncode == 0:
                print("Initializing Git LFS and pulling large model/dataset assets...")
                run_command("git lfs install")
                run_command("git lfs pull")
                print_success("Git LFS assets successfully pulled!")
            else:
                print_warning("Git LFS is not installed on your system. Please install git-lfs via your system package manager (e.g., 'sudo apt install git-lfs') to download large datasets/models.")
        except Exception as e:
            print_warning(f"Skipping Git LFS setup: {e}")
    else:
        print_warning("Git command not found. Skipping Git LFS configuration.")

    # Step 5: Verification Diagnostics
    print_step("Step 5: Running Verification Diagnostics")
    imports_to_test = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("ipywidgets", "IPyWidgets (GUI)"),
        ("lerobot", "LeRobot Core Library")
    ]
    
    all_passed = True
    for module_name, display_name in imports_to_test:
        try:
            # Run import check in a subprocess using the current environment's Python
            subprocess.run([sys.executable, "-c", f"import {module_name}"], check=True, capture_output=True)
            print_success(f"{display_name} ('{module_name}') is functional.")
        except subprocess.CalledProcessError:
            print_error(f"{display_name} ('{module_name}') failed to import!")
            all_passed = False

    print("\n====================================================")
    if all_passed:
        print_success("LeRobot Environment is 100% READY! 🎉")
        print("\nYou can now open Jupyter Notebook and run your tasks:")
        print("  jupyter notebook lhwdev/imitation-learning.ipynb")
    else:
        print_warning("Environment setup completed, but some verification diagnostics failed.")
        print("Please check the errors above before running the notebook.")
    print("====================================================\n")

if __name__ == "__main__":
    main()
