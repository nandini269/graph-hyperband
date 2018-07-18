# Hyperband for Graph Neural Networks
This repository contains graph convolutional networks (or message passing network) for molecule property prediction. 

## Training (Classification task)
```
mkdir model
python main.py --train $TRAIN_FILE --valid $VALID_FILE --test $TEST_FILE --save_dir model --metric classify
```
This script will train the best network with at most 40 epochs, and save the best model in `model/model.best`.
It will then test the best model and give you the test error.
The input file `TRAIN_FILE` has to be a CSV file with a header row.

The above code assumes the task is binary classification.

## Training (Regression task)
```
mkdir model
python main.py --train $TRAIN_FILE --valid $VALID_FILE --test $TEST_FILE --save_dir model --metric regress
```

