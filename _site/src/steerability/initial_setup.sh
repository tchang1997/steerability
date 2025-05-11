#!/bin/bash
echo -n "Making results directories..."
mkdir -p results/steerability
mkdir -p cache/goalspace
echo -n "Downloading auxiliary dependencies..."
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('wordnet')"
