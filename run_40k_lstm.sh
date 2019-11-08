#!/usr/bin/env bash
FOLDER=40k
NAME=lstm
mkdir results_${FOLDER}
mkdir models_${FOLDER}

for SEED in 12093 12345 45678
do
python3 -u -m equation_verification.nn_tree_experiment \
    --seed $SEED \
    --train-path data/40k_train.json \
    --validation-path data/40k_val_shallow.json \
    --test-path data/40k_test.json \
    --model-class LSTMTrees \
    --checkpoint-every-n-epochs 1 \
    --result-path results_${FOLDER}/exp_${NAME}_${SEED}.json \
    --model-prefix models_${FOLDER}/exp_${NAME}_${SEED} > results_${FOLDER}/exp_${NAME}_${SEED}.log &
done