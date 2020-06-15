# Memory Augmented Recursive Neural Networks

This is the code for the Memory [Tree Stack Memory Units](https://arxiv.org/abs/1911.01545).

## Dependencies
Python 3.6 or higher

PyTorch 1.0 or higher

## Visualization tool
With this tool you can explore the learned representations at the output of each node, the elements of the stack and the learned push and pop operations for each equation from train, test or validation data. Read the [using_explore_trace.txt](https://github.com/ForoughA/recursiveMemNet/blob/master/using_explore_trace.txt) file to understand how to use the tool. Two example outputs of the visualization tool are stored in folder [visualization/](https://github.com/ForoughA/recursiveMemNet/tree/master/visualization)

## Navigation
1. data/ 
    * is a folder containing the data used in all the experiments reported in the paper. we have included the train/test/validation splits that were used in the paper. There are two sets of data for each of the experiments in the paper
        - Equation Verification
            - code/data/40k_test.json 
            - code/data/40k_train.json
            - code/data/40k_val_shallow.json
        - Equation Completion
            - code/data/blanks/40k_test_blank_{i}.json for i in range 0 through 14. (test data for equation completion. This is the data from data/40k_test.json in which sub-trees of depth 1 and 2 are randomly replaced with blanks)
            - code/data/candidate_classes.json (contain equvalence classes for the blank candidates to compute top-k accuracy. Expressions in the same class evaluate to the same value. For example, 1+1 and 2 are in the same class. if the top ranked candidate belongs to the same class as the correct answer, then it is considered a correct blank prediction.)
2. equation_verification/
    * is a folder containing the python scripts implementing the proposed model. There are 5 filed in this folder.
        - equation_verification/__init__.py
        - equation_verification/constants.py
        - equation_verification/dataset_loading.py
        - equation_verification/nn_tree_experiment.py
        - code/equation_verification/nn_tree_model.py (implementation of Tree-SMU, Tree-LSTM and Tree-RNN)
        - code/equation_verification/equation_completion_experiment.py
        - code/equation_verification/sequential_model.py (recurrent neural network (RNN) and LSTM implementation)
        - code/equation_verification/sequential_model_constants.py
3. Shell scripts can be used to replicate the experiments in the paper. These are:
    * Equation Verification Experiments (all the hyperparameters are default settings and not the optimal hyperparameters.)
        - code/run_40k_gatedpushpop_normalize_no_op.sh that replicates the Tree-SMU results. Other model abblations can be replicatd by setting the corresponding command line arg in this script by removing the normalize and the no-op flags.
        - code/run_40k_lstm.sh that replicates tree-LSTM results 
        - code/run_40k_vanilla.sh that replicatd tree-RNN results 
    * Equation Completion Experiments
        - code/complete_40k.sh that replicates tree-SMU, tree-LSTM and tree-RNN results using the trained model (best seed and best hyper-parameters) for Tree-SMU
        - code/aggregate_test_splits.py this is a python script that aggregates the results of code/complete_40k.sh into a single json. Usage is:
            * python3 aggregate_test_splits.py results_completion  ${prefix}
            * prefix is the prefix of the output of data/complete_40k.sh. For example: exp_gatedpushpop_normalize_no_op_completion_50_0.1_12345_97_
            * the output will be written to : ${prefix}agg.json in the same folder (results_completion ) 
4. code/checkpoints/ 
    * This folder contains the model checkpoints for the best trained models for equation verification and equation completion picked based on the maximum accuracy on the validation data.
5. visualization tool that can be used to visualize a learned tree model for any chosen input equation and explore the learned weights and stack operations. Relevant files are:
    - using_explore_trace.txt is a text file that contains instructions for how to use the visualization tool
    - visualization/ the visualization results will be saved in this folder. We have included two visualization examples in this folder just to give a feeling of what to expect. 
    - explore_trace.py python script. There is no need to run this script, we have provided shell scripts for that.
    - explore_trace.sh shell script for running the visualization tool. Please refer to using_explore_trace.txt to see how to use this
    - trace_40k_gatedpushpop_normalize_no_op.sh shell script for running the visualization tool. Please refer to using_explore_trace.txt to see how to use this
    - parse.py parses strings into our the tree class in our python code
    - optimizers.py are model optimizers: adam and sgd

## Notebooks
stay tuned for IPython (Jupyter) Notebooks...
