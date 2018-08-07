# Hyperband for Graph Neural Networks
This repository contains code for automatically optimizing hyper-parameters for graph convolutional networks (or message passing networks). It uses the hyperband algorithm. 

The hyperband algorithm takes in two parameters - eta and max_iter. Eta is the proportion by which the algorithm carries out successive halving and max_iter is the maximum number of iterations a dataset can be trained on. Hyperband tries an optimized version of random search based on these two parameters.

## Training (Classification task)
```
mkdir model
python main.py --train $TRAIN_FILE --metric classify --save_dir model
```
Here the first argument takes in a path to the data file. The code splits this into training, validation and test sets. This script will train the best network with at most 40 epochs, and save the best model in `model/model.best`.
It will then test the best model and give you the test error.
The input file `TRAIN_FILE` has to be a CSV file with a header row.

The above code assumes the task is binary classification.

## Training (Regression task)
```
mkdir model
python main.py --train $TRAIN_FILE --metric regress --save_dir model
```

