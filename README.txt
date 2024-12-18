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

4. Code Explanation
4.1 “Bayes_for_HNN_gasoline_FCC.py” 
This file is the code for using Bayesian algorithm to optimize the hyperparameters of neural network in the lab-scale hybrid model. The input (features) are process conditions and naphtha molecular composition of lab-scale reactor, and the label is the product molecular composition of lab-scale reactor.
4.2 “HNN_gasoline_FCC_lab_train.py”
This file is a training code for naphtha catalytic cracking lab-scale model by tuned hyperparameters. The input (features) are process conditions and naphtha molecular composition of lab-scale reactor, and the label is the product molecular composition of lab-scale reactor.
4.3 “HNN_gasoline_FCC_lab_pred.py”
This file is a testing code for naphtha catalytic cracking lab-scale model. Through loading the trained neural network, the code can predict the product molecular composition by inputting process conditions and naphtha molecular composition of lab-scale reactor.
4.4 “Transfer_learning_Bayes_opt_para_for_HNN_prop.py”
This file is the code for optimizing hyperparameters in the transfer learning using the Bayesian algorithm. The input (features) are process conditions and naphtha molecular composition of pilot-scale reactor, and the label is the product bulk properties of pilot-scale reactor.
4.5 “HNN_gasoline_FCC_pilot_train.py”
This file is a training code for naphtha catalytic cracking pilot-scale model by tuned hyperparameters. The input (features) are process conditions and naphtha molecular composition of pilot-scale reactor, and the label is the product bulk properties of pilot-scale reactor.
4.6 “HNN_gasoline_FCC_pilot_pred.py”
This file is a testing code for naphtha catalytic cracking pilot-scale model. Through loading the trained pilot-scale hybrid model, the code can predict the product bulk properties by inputting process conditions and naphtha molecular composition of pilot-scale reactor.
5. Dataset Explanation
5.1 “FeedMoleculeContent_lab.csv”
This file is the input data of the naphtha catalytic cracking lab-scale model, including process conditions and naphtha molecular composition. The first four columns are process conditions, and the last 129 columns are naphtha composition.
5.2 “ProductMoleculeContent_lab.csv”
This file is the output data of the naphtha catalytic cracking lab-scale model, including product molecular composition of 129 molecules.
5.3 “FeedMoleculeContent_pilot.csv”
This file is the input data of the naphtha catalytic cracking pilot-scale model, including process conditions and naphtha molecular composition. The first four columns are process conditions, and the last 129 columns are naphtha composition.
5.4 “ProductBulkProperty_pilot.csv”
This file is the output data of the naphtha catalytic cracking pilot-scale model, including product bulk property data.

6. Other File Explanation
6.1 “model_checkpoint_net1.csv”
This file is the trained Process-based ResNet for lab-scale hybrid model.
6.2 “model_checkpoint_net2.csv”
This file is the trained Molecule-based ResNet for lab-scale hybrid model.

6.3 “model_checkpoint_agg_net.csv”
This file is the trained integrated ResNet for lab-scale hybrid model.
6.4 “model_checkpoint_net1_prop_60.csv”
This file is the trained Process-based ResNet for pilot-scale hybrid model, when the pilot-scale dataset is 60.
6.5 “model_checkpoint_net2_prop_60.csv”
This file is the trained Molecule-based ResNet for pilot-scale hybrid model, when the pilot-scale dataset is 60.
6.6 “model_checkpoint_agg_net_prop_60.csv”
This file is the trained integrated ResNet for pilot-scale hybrid model, when the pilot-scale dataset is 60.


