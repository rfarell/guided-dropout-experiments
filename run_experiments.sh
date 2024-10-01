#!/bin/bash

# Exit if any command fails
set -e

# Step 1: Create required directories if they don't exist
echo "Creating necessary directories..."
mkdir -p trained_models
mkdir -p plots
mkdir -p data

# Step 2: Install required Python packages
echo "Installing required packages..."
pip install -r requirements.txt

# Step 3: Train the models

# Train Guided Dropout Model
echo "Training Guided Dropout Model..."
python train.py --model guided --epochs 50

# Train Regular Dropout Model
echo "Training Regular Dropout Model..."
python train.py --model regular --epochs 50

# Train No Dropout Model
echo "Training No Dropout Model..."
python train.py --model none --epochs 50

# Step 4: Visualize activation strengths
echo "Visualizing activation strengths..."
python visualize_activation_strengths.py

# Step 5: Notify user of successful completion
echo "Experiment completed successfully!"
echo "Trained models are saved in 'trained_models' directory."
echo "Plots are saved in 'plots' directory."
