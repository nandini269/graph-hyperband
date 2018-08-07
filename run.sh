#!/bin/bash

ROOT="/scratch4/nandini_nayar/hyperband2"
RESULTS="$ROOT/$1"

mkdir -p $RESULTS

#Small datasets
echo "training delaney"
mkdir -p $RESULTS/delaney
python main.py --train data/delaney.csv --metric regress --save_dir model | tee $RESULTS/delaney/LOG1
python main.py --train data/delaney.csv --metric regress --save_dir model | tee $RESULTS/delaney/LOG2

echo "training freesolv"
mkdir -p $RESULTS/freesolv
python main.py --train data/freesolv.csv --metric regress --save_dir model | tee $RESULTS/freesolv/LOG1
python main.py --train data/freesolv.csv --metric regress --save_dir model | tee $RESULTS/freesolv/LOG2

echo "training lipo"
mkdir -p $RESULTS/lipo
python main.py --train data/lipo.csv --metric regress --save_dir model | tee $RESULTS/lipo/LOG1
python main.py --train data/lipo.csv --metric regress --save_dir model | tee $RESULTS/lipo/LOG2

echo "training bace"
mkdir -p $RESULTS/bace
python main.py --train data/bace.csv --metric classify --save_dir model --split scaffold | tee $RESULTS/bace/LOG1
python main.py --train data/bace.csv --metric classify --save_dir model --split scaffold | tee $RESULTS/bace/LOG2

echo "training BBBP"
mkdir -p $RESULTS/BBBP
python main.py --train data/BBBP.csv --metric classify --save_dir model --split scaffold | tee $RESULTS/BBBP/LOG1
python main.py --train data/BBBP.csv --metric classify --save_dir model --split scaffold | tee $RESULTS/BBBP/LOG2

echo "training tox21"
mkdir -p $RESULTS/tox21
python main.py --train data/tox21.csv --metric classify --save_dir model | tee $RESULTS/tox21/LOG1
#python main.py --train data/tox21.csv --metric classify --save_dir model | tee $RESULTS/tox21/LOG2

echo "training sider"
mkdir -p $RESULTS/sider
python main.py --train data/sider.csv --metric classify --save_dir model | tee $RESULTS/sider/LOG1
python main.py --train data/sider.csv --metric classify --save_dir model | tee $RESULTS/sider/LOG2

