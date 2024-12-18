# 1. Overview
This is a code for accelerating the scale-up of complex reaction systems by integrating the mechanistic model with deep transfer learning (hybrid model). The code includes a lab-scale hybrid model and a pilot-scale hybrid model for naphtha catalytic cracking. Moreover, a multi-objective optimization algorithm coupling the deep neural network was provided. The code can download by GitHub, and run in python package.
# 2. System Requirements
2.1 Hardware requirements
The hybrid model requires only a standard computer with enough RAM to support the in-memory operations.
# 3. Software requirements
## 3.1 OS Requirements
The hybrid model can be performed by *Windows*. The model has been tested on the following systems:
+ Windows 10 and 11
## 3.2 Python Dependencies
The hybrid model mainly depends on the Python scientific stack.
```
torch
optuna
numpy
sklearn
pandas
matplotlib
joblib
pymoo
```
The hybrid model has been tested on the following python environments:
+ Python 3.9
+ Pytorch 2.0.0+cu118
# 4. Code and dataset
The entire model is divided into three parts, namely, lab-scale model, pilot-scale model and process optimization, which are placed in three folders respectively.
## 4.1 lab scale model
Code section
### 4.1.1 “Bayes_for_HNN_gasoline_FCC.py” 
This file is the code for using Bayesian algorithm to optimize the hyperparameters of neural network in the lab-scale hybrid model. The input (features) are process conditions and naphtha molecular composition of lab-scale reactor, and the label is the product molecular composition of lab-scale reactor.
### 4.1.2 “HNN_gasoline_FCC_lab_train.py”
This file is a training code for naphtha catalytic cracking lab-scale model by tuned hyperparameters. The input (features) are process conditions and naphtha molecular composition of lab-scale reactor, and the label is the product molecular composition of lab-scale reactor.
### 4.1.3 “HNN_gasoline_FCC_lab_pred.py”
This file is a testing code for naphtha catalytic cracking lab-scale model. Through loading the trained neural network, the code can predict the product molecular composition by inputting process conditions and naphtha molecular composition of lab-scale reactor.
