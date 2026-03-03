## Project Overview

This project addresses the problem of RNA Binding Protein (RBP)–RNA interaction prediction. RNA Binding Proteins (RBPs) interact with RNA molecules by recognizing specific sequence motifs or structural patterns, and the strength of this interaction (referred to as binding affinity) is a continuous value.  

In this project, the task is formulated as a regression problem, where deep neural networks are trained to predict the binding affinity of RNA sequences for the RBP RBFOX1, using data derived from the RNAcompete experimental protocol. To address this problem, two different deep neural network models are implemented and compared. The models are built from scratch using PyTorch.

This project was developed in the context of the course of Deep Learning.


## Instructions to run experiment
### Exercise 2
Make sure you are inside the `Q2` directory before running any scripts:

* **BiLSTM Model:** To train and test the BiLSTM model and get the results, run `python3 run_bilstm.py`
* **CNN Model:** To train and test the CNN model and get the results, run `python3 run_cnn.py`
