#!/usr/bin/env bash
FOLDER=checkpoints
RESULTS=results_completion
mkdir ${RESULTS}

for DATASPLIT in {0..14}
do
    NAME=vanilla \
    SEED=45678 \
    EPOCH=66 \
    NUM_HIDDEN=45 \
    DROPOUT=0.1 \
    && \
        python3 -u -m equation_verification.equation_completion_experiment \
            --seed $SEED \
            --train-path 40k_empty.json \
            --validation-path 40k_empty.json \
            --test-path data/blanks/40k_test_blank_${DATASPLIT}.json \
            --candidate-path data/candidate_classes.json \
            --model-class NNTrees \
            --dropout ${DROPOUT} \
            --num-hidden ${NUM_HIDDEN} \
            --checkpoint-every-n-epochs 1 \
            --result-path ${RESULTS}/exp_${NAME}_completion_${NUM_HIDDEN}_${DROPOUT}_${SEED}_${EPOCH}_${DATASPLIT}.json \
            --model-prefix ${FOLDER}/exp_${NAME}_drop${DROPOUT}_hidden${NUM_HIDDEN}_${SEED} \
            --load-epoch ${EPOCH} \
            --evaluate-only \
            --cut 1000000 > ${RESULTS}/exp_${NAME}_completion_${NUM_HIDDEN}_${DROPOUT}_${SEED}_${EPOCH}_${DATASPLIT}.log &
done

for DATASPLIT in {0..14}
do
    NAME=lstm \
    EPOCH=68 \
    NUM_HIDDEN=50 \
    DROPOUT=0.1 \
    SEED=12093 \
    && \
        python3 -u -m equation_verification.equation_completion_experiment \
            --seed $SEED \
            --train-path 40k_empty.json \
            --validation-path 40k_empty.json \
            --test-path data/blanks/40k_test_blank_${DATASPLIT}.json \
            --candidate-path data/candidate_classes.json \
            --model-class LSTMTrees \
            --num-hidden ${NUM_HIDDEN} \
            --dropout ${DROPOUT} \
            --checkpoint-every-n-epochs 1 \
            --result-path ${RESULTS}/exp_${NAME}_completion_${NUM_HIDDEN}_${DROPOUT}_${SEED}_${EPOCH}_${DATASPLIT}.json \
            --model-prefix ${FOLDER}/exp_${NAME}_drop${DROPOUT}_hidden${NUM_HIDDEN}_${SEED} \
            --load-epoch ${EPOCH} \
            --evaluate-only \
            --cut 1000000 > ${RESULTS}/exp_${NAME}_completion_${NUM_HIDDEN}_${DROPOUT}_${SEED}_${EPOCH}_${DATASPLIT}.log &
done

for DATASPLIT in {0..14}
do
    NAME=gatedpushpop_normalize_no_op \
    SEED=12345 \
    EPOCH=97 \
    NUM_HIDDEN=50 \
    DROPOUT=0.1 \
    && \
        python3 -u -m equation_verification.equation_completion_experiment \
            --seed $SEED \
            --train-path 40k_empty.json \
            --validation-path 40k_empty.json \
            --test-path data/blanks/40k_test_blank_${DATASPLIT}.json \
            --candidate-path data/candidate_classes.json \
            --stack-node-activation tanh \
            --tree-node-activation tanh \
            --model-class StackNNTreesMem2out \
            --num-hidden ${NUM_HIDDEN} \
            --dropout ${DROPOUT} \
            --top-k 1 \
            --stack-type stack \
            --gate-push-pop \
            --normalize-action \
            --no-op \
            --checkpoint-every-n-epochs 1 \
            --result-path ${RESULTS}/exp_${NAME}_completion_${NUM_HIDDEN}_${DROPOUT}_${SEED}_${EPOCH}_${DATASPLIT}.json \
            --model-prefix ${FOLDER}/exp_${NAME}_${NUM_HIDDEN}_${DROPOUT}_${SEED} \
            --load-epoch ${EPOCH} \
            --evaluate-only \
            --cut 1000000 > ${RESULTS}/exp_${NAME}_completion_${NUM_HIDDEN}_${DROPOUT}_${SEED}_${EPOCH}_${DATASPLIT}.log &
done