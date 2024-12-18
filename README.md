1. Overview
This is a code for accelerating the scale-up of complex reaction systems by integrating the mechanistic model with deep transfer learning (hybrid model). The code includes a lab-scale hybrid model and a pilot-scale hybrid model for naphtha catalytic cracking. Moreover, a multi-objective optimization algorithm coupling the deep neural network was provided. The code can download by GitHub, and run in python package.
2. System Requirements
2.1 Hardware requirements
The hybrid model requires only a standard computer with enough RAM to support the in-memory operations.
3. Software requirements
3.1 OS Requirements
The hybrid model can be performed by *Windows*. The model has been tested on the following systems:
+ Windows 10 and 11
3.2 Python Dependencies
The hybrid model mainly depends on the Python scientific stack.
```
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
4. Code and dataset
