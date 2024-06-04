#!/bin/bash
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz 
tar xvzf gdb9.tar.gz
rm gdb9.tar.gz

wget https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
wget https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz

wget https://raw.githubusercontent.com/gablg1/ORGAN/master/data/qm9_5k.csv
tail -n +2 "qm9_5k.csv" > "qm9_5k.smi"
rm "qm9_5k.csv"
