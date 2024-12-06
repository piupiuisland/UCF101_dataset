#!/bin/bash

# Define variables
URL="https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
OUTPUT_DIR="./"  # Directory to store the extracted data
RAR_FILE="UCF101.rar"

# Step 1: Download the file
echo "Downloading UCF101 dataset..."
wget --no-check-certificate -O $RAR_FILE $URL  # Added --no-check-certificate

# Step 2: Check if unrar is installed
if ! command -v unrar &> /dev/null
then
    echo "unrar not found. Installing unrar..."
    sudo apt update && sudo apt install -y unrar
fi

# Step 3: Extract the .rar file
echo "Extracting UCF101 dataset..."
unrar x $RAR_FILE $OUTPUT_DIR

# Step 4: Clean up
echo "Cleaning up..."
rm $RAR_FILE

echo "UCF101 dataset downloaded and extracted to $OUTPUT_DIR."
