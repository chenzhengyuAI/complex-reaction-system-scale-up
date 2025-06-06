# 1. Overview
This is a code for accelerating the scale-up of complex reaction systems by integrating the mechanistic model with deep transfer learning (hybrid model). The code includes a lab-scale hybrid model and a pilot-scale hybrid model for naphtha catalytic cracking. Moreover, a multi-objective optimization algorithm coupling the deep neural network was provided. The code can download by GitHub, and run in python package.
# 2. System Requirements
2.1 Hardware requirements
The hybrid model requires only a standard computer with enough RAM to support the in-memory operations.
## 2.1 Software requirements
### 2.1.1 OS Requirements
The hybrid model can be performed by *Windows*. The model has been tested on the following systems:
+ Windows 10 and 11
### 2.1.2 Python Dependencies
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
+ torch 2.0.0+cu118
# 3. Installation Guide
Download the above model to your local computer, modify the file path in the corresponding code, and then run the code
+ Computational time for Bayesian hyperparameter optimization: 8 hours
+ Computational time for model train: 10 min
+ Computational time for model prediction: < 20 s
# 4. Code and Dataset
The entire model is divided into three parts, namely, lab-scale model, pilot-scale model and process optimization, which are placed in three folders respectively.
## 4.1 Lab scale model
### 4.1.1 Code section
+ “Bayes_for_HNN_gasoline_FCC.py”: 
This file is the code for using Bayesian algorithm to optimize the hyperparameters of neural network in the lab-scale hybrid model. The input are process conditions and naphtha molecular composition of lab-scale reactor (FeedMoleculeContent_lab.csv), and the label is the product molecular composition of lab-scale reactor (ProductMoleculeContent_lab.csv).
+ “HNN_gasoline_FCC_lab_train.py”:
This file is a training code for naphtha catalytic cracking lab-scale model by tuned hyperparameters. The input are process conditions and naphtha molecular composition of lab-scale reactor (FeedMoleculeContent_lab.csv), and the label is the product molecular composition (ProductMoleculeContent_lab.csv) of lab-scale reactor (ProductMoleculeContent_lab.csv).
+ “HNN_gasoline_FCC_lab_pred.py”:
This file is a testing code for naphtha catalytic cracking lab-scale model. Through loading the trained neural network, the code can predict the product molecular composition (targets_train_lab.csv/targets_test_lab.csv) by inputting process conditions and naphtha molecular composition of lab-scale reactor (input_train_lab.csv/input_test_lab.csv).
### 4.1.2 Dataset section
+ “FeedMoleculeContent_lab.csv”:
This file is the input data of the naphtha catalytic cracking lab-scale model, including process conditions and naphtha molecular composition. The first four columns are process conditions, and the last 129 columns are naphtha composition.
+ “ProductMoleculeContent_lab.csv”:
This file is the output data of the naphtha catalytic cracking lab-scale model, including product molecular composition of 129 molecules.
+ “input_train_lab”:
This file is the input data of train dataset for lab scale reactor
+ “targets_train_lab”:
This file is the output data of train dataset for lab scale reactor
+ “input_test_lab”:
This file is the input data of train dataset for lab scale reactor
+ “targets_test_lab”:
This file is the output data of train dataset for lab scale reactor
## 4.2 Pilot scale model
### 4.2.1 Code section
+ “Transfer_learning_Bayes_opt_para_for_HNN_prop.py”:
This file is the code for optimizing hyperparameters in the transfer learning using the Bayesian algorithm. The input are process conditions and naphtha molecular composition of pilot-scale reactor (FeedMoleculeContent_pilot.csv), and the label is the product bulk properties of pilot-scale reactor (ProductBulkProperty_pilot.csv).
+ “HNN_gasoline_FCC_pilot_train.py”:
This file is a training code for naphtha catalytic cracking pilot-scale model by tuned hyperparameters. The input (features) are process conditions and naphtha molecular composition of pilot-scale reactor (FeedMoleculeContent_pilot.csv), and the label is the product bulk properties of pilot-scale reactor (ProductBulkProperty_pilot.csv).
+ “HNN_gasoline_FCC_pilot_pred.py”:
This file is a testing code for naphtha catalytic cracking pilot-scale model. Through loading the trained pilot-scale hybrid model, the code can predict the product bulk properties (targets_train_pilot.csv/targets_test_pilot.csv) by inputting process conditions and naphtha molecular composition of pilot-scale reactor (input_train_pilot.csv/input_train_pilot.csv).
### 4.2.2 Dataset section
+ “FeedMoleculeContent_pilot.csv”:
This file is the input data of the naphtha catalytic cracking pilot-scale model, including process conditions and naphtha molecular composition. The first four columns are process conditions, and the last 129 columns are naphtha composition.
+ “ProductBulkProperty_pilot.csv”:
This file is the output data of the naphtha catalytic cracking pilot-scale model, including product bulk property data.
+ “input_train_pilot”:
This file is the input data of train dataset for pilot plant
+ “targets_train_pilot”:
This file is the output data of train dataset for pilot plant
+ “input_test_pilot”:
This file is the input data of train dataset for pilot plant
+ “targets_test_pilot”:
This file is the output data of train dataset for pilot plant
## 4.3 Process optimization model
### 4.3.1 Code section
+ “GA_optimization_HNN_gasoline_FCC_pilot.py”:
This file is the optimization algorithm by GA coupled with hybrid model.
# 5. Other File Explanation
+ “model_checkpoint_net1.csv”:
This file is the trained Process-based ResNet for lab-scale hybrid model.
+ “model_checkpoint_net2.csv”:
This file is the trained Molecule-based ResNet for lab-scale hybrid model.
+ “model_checkpoint_agg_net.csv”:
This file is the trained integrated ResNet for lab-scale hybrid model.
+ “model_checkpoint_net1_prop_60.csv”:
This file is the trained Process-based ResNet for pilot-scale hybrid model, when the pilot-scale dataset is 60.
+ “model_checkpoint_net2_prop_60.csv”:
This file is the trained Molecule-based ResNet for pilot-scale hybrid model, when the pilot-scale dataset is 60.
+ “model_checkpoint_agg_net_prop_60.csv”:
This file is the trained integrated ResNet for pilot-scale hybrid model, when the pilot-scale dataset is 60.
# 6. Model Result Validation
## 6.1 Laboratory-scale model validation
If one intend to validate the lab-scale model, please run the script “HNN_gasoline_FCC_lab_pred.py”.
Lines 93–94: Inputting “input_train_lab.csv” and “targets_train_lab.csv” will generate results for the laboratory-scale model training set.
Lines 93–94: Inputting “input_test_lab.csv” and “targets_test_lab.csv” will generate results for the laboratory-scale model test set.
## 6.2 Pilot-scale model validation
If one intend to validate the pilot-scale model, please run the script “HNN_gasoline_FCC_pilot_pred_prop.py”.
Lines 104–105: Inputting “input_train_pilot.csv” and “targets_train_pilot.csv” will generate results for the pilot-scale model training set.
Lines 104–105: Inputting “input_validation_pilot.csv” and “targets_validation_pilot.csv” will generate results for the pilot-scale validation set.
Lines 104–105: Inputting “input_test_pilot.csv” and “targets_test_pilot.csv” will generate results for the pilot-scale testing set.
## 6.3 Process ptimization validation
If one intend to validate the process optimization results, please run the script “GA_optimization_HNN_gasoline_FCC_pilot.py”.
# 7. Model Interface Usage
We programmed a web version of the interface based on Streamlit. The code to run the interface is uploaded to Git Hub. “APP_lab_model.py” in the “lab scale model” folder is an interface for the laboratory-scale model, while “APP_pilot_model.py” in the “pilot scale model” folder is an interface for the pilot-scale model. 
For example, if one wants to run the “APP_lab_model.py”, one can execute the command “streamlit run APP_lab_model.py” in the Python environment to open the corresponding web page. The left side of the interface is used for data input, such as “input_train_lab.csv” and targets_train_lab.csv”. The right side of the interface shows the model training results.
Similarly, if one wants to run the “APP_pilot_model.py”, one can execute the command “streamlit run APP_pilot_model.py” in the Python environment to open the corresponding web page. “input_train_lab.csv” and “targets_train_lab.csv” should be input on the left side of the interface, and the calculated results will be presented on the right side of the interface.
