# Memory Augmented Recursive Neural Networks

This is the code for the Memory [Augmented Recursive Neural Networks](https://arxiv.org/abs/1911.01545).

## Dependencies
Python 3.6 or higher
PyTorch 1.0 or higher

## Visualization tool
With this tool you can explore the learned representations at the output of each node, the elements of the stack and the learned push and pop operations for each equation from train, test or validation data. Read the [using_explore_trace.txt](https://github.com/ForoughA/recursiveMemNet/blob/master/using_explore_trace.txt) file to understand how to use the tool. Two example outputs of the visualization tool are stored in folder visualization/

## Navigation
1. data/ 
    * is a folder containing the data used in all the experiments reported in the paper. we have included the train/test/validation splits that were used in the paper. There are three files in this folder. 
        - data/40k_test.json
        - data/40k_train.json
        - data/40k_val_shallow.json
2. equation_verification/
    * is a folder containing the python scripts implementing the proposed model. There are 5 filed in this folder.
        - equation_verification/__init__.py
        - equation_verification/constants.py
        - equation_verification/dataset_loading.py
        - equation_verification/nn_tree_experiment.py
        - equation_verification/nn_tree_model.py
3. Shell scripts can be used to replicate the experiments in the paper. These are:
    - run_40k_lstm.sh that replicates tree-LSTM results
    - run_40k_vanilla.sh that replicatd tree-RNN results
    - run_40k_gatedpushpop_normalize_no_op.sh that replicates the best stack augmented tree-LSTM + normalize + no_op experiment. Other model abblations can be replicatd by setting the corresponding command line arg in this script.

4. visualization tool that can be used to visualize a learned tree model for any chosen input equation and explore the learned weights and stack operations. Relevant files are:
    - using_explore_trace.txt is a text file that contains instructions for how to use the visualization tool
    - visualization/ the visualization results will be saved in this folder. We have included two visualization examples in this folder just to give a feeling of what to expect. 
    - explore_trace.py python script. There is no need to run this script, we have provided shell scripts for that.
    - explore_trace.sh shell script for running the visualization tool. Please refer to using_explore_trace.txt to see how to use this
    - trace_40k_gatedpushpop_normalize_no_op.sh shell script for running the visualization tool. Please refer to using_explore_trace.txt to see how to use this
    - parse.py parses strings into our the tree class in our python code
    - optimizers.py are model optimizers: adam and sgd

## Notebooks
stay tuned for IPython (Jupyter) Notebooks...
