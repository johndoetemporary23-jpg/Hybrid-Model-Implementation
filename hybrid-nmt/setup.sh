#!/bin/bash
set -e

echo "=================================================================================="
echo "Hybrid NMT Setup: IndicBERT v2 + IndicTrans2 for English → Meitei Translation"
echo "=================================================================================="

# Step 1: Install Python dependencies
echo "Step 1: Installing Python dependencies..."
pip install -q --upgrade pip setuptools wheel
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Step 2: Download IndicBERT v2
echo "Step 2: Downloading IndicBERT v2 (ai4bharat/IndicBERTv2-MLM-Sam-TLM)..."
huggingface-cli download ai4bharat/IndicBERTv2-MLM-Sam-TLM --repo-type model --local-dir ./models/IndicBERT-v2 2>&1 | grep -E "^(Downloading|✓)" || true
echo "✓ IndicBERT v2 downloaded"

# Step 3: Download IndicTrans2
echo "Step 3: Downloading IndicTrans2 (ai4bharat/indictrans2-en-indic-dist-200M)..."
huggingface-cli download ai4bharat/indictrans2-en-indic-dist-200M --repo-type model --local-dir ./models/IndicTrans2 2>&1 | grep -E "^(Downloading|✓)" || true
echo "✓ IndicTrans2 downloaded"

# Step 4: Install IndicTransToolkit
echo "Step 4: Installing IndicTransToolkit from GitHub..."
pip install -q git+https://github.com/VarunGumma/IndicTransToolkit.git
echo "✓ IndicTransToolkit installed"

# Step 5: Verify setup
echo ""
echo "=================================================================================="
echo "Setup Complete!"
echo "=================================================================================="
echo ""
echo "Next steps:"
echo "1. Run data preprocessing: python data/preprocess.py"
echo "2. Start training: python train.py"
echo ""
