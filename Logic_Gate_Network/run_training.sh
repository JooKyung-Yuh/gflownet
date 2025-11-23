#!/bin/bash
# Quick training script - runs existing tests with different iterations

echo "================================================================================"
echo "Logic Gate Network GFlowNet - Training"
echo "================================================================================"
echo ""
echo "Running training with 5 iterations (quick test)..."
python -m pytest tests/test_gnn_training.py::test_train_multiple_steps -v

echo ""
echo "================================================================================"
echo "Training complete!"
echo "================================================================================"
