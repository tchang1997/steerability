#!/bin/bash
echo "Initializing directories..."
mkdir -p cache
mkdir -p data
mkdir -p logs
mkdir -p results/raw
mkdir -p results/judged
mkdir -p results/steerability_metrics
echo "Downloading required models..."
python -m spacy download en_core_web_sm
echo "Downloading steerability probes..."
huggingface-cli download --repo-type dataset XXXX --local-dir ./data
