#!/bin/bash
set -e

echo "=== Image Context Compression Experiment ==="
echo ""

# 1. Install dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

# 2. Run smoke test
echo ""
echo "Step 2: Running smoke test..."
python tests/test_smoke.py

# 3. Run small experiment (optional, uncomment to run)
# echo ""
# echo "Step 3: Running smoke test experiment..."
# python scripts/run_exp.py --config configs/smoke_test.yaml

# 4. Generate plots (if results exist)
# echo ""
# echo "Step 4: Generating plots..."
# if [ -f "outputs/results/smoke_results.jsonl" ]; then
#     python scripts/plot_results.py --input outputs/results/smoke_results.jsonl --output outputs/plots/
# fi

echo ""
echo "=== Setup complete! ==="
echo "To run full experiment:"
echo "  python scripts/run_exp.py --config configs/exp_sweep.yaml"
echo ""
echo "To generate plots:"
echo "  python scripts/plot_results.py --input outputs/results/results.jsonl --output outputs/plots/"
