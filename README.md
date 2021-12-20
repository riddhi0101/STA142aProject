# STA142aProject

This was a group project for a statistical learning class focused on supervised methods. A brief summary of the project is described below. 


Drug Classification
Riddhi Barbhaiya, Rohit Haritsa, Helen Le

Project Description/Summary
The goal of this project is to predict consumption of various drugs(specified below) using personality and demographic information. We treat this as a multi-label classification problem. To solve this classification we explore two methods:
	1) One vs All classification: we build five classifiers - one for each drug of interest
	2) Neural Network: we build a neural network that predicts all 5 drugs simultaneously
Strategy one does not consider any relations between the drugs which may be informative in classification. On the other hand, due to the dataset being fairly small, the neural network is more likely to overfit. We measure the performance of the two methods by looking at global measures such as hamming loss and subset accuracy. We also evaluate measures per class such as AUC, recall, and specificity.

Dataset
We use the drug consumption (quantified) dataset from the UCI Machine Learning Repository. The dataset includes 12 predictor variables — age, gender, education, country, ethnicity, 7 personality measures. Many of the predictor variables are categorical. The dataset that we use has quantified these variables using ordinal and nominal feature combination and sparse PCA (Fehrman et al, 2017). We focus on predicting usage of five drugs: cocaine, LSD, heroin, benzodiazepine (benzos), and caffeine. Predicting drug consumption can be applicable in predicting abuse and addiction. Demographic information and personality information are easy to obtain and would be a feasible to use to predict abuse.

Results Summary
We are able to classify each drug with over 70\% accuracy (table 2). Both methods perform comparably. This indicates that there isn’t really an advantage to employing a complicated method(neural network) when the simpler method performs closely.


Files in this repo:


- NeuralNet folder contains code to the training and validation of the neural network.
	- NNFunctions.py: general training and testing functions implemented using PyTorch
	- HyperparameterSelection.ipynb: selecting hyper parameters
	- FinalNN: the training for the network with hyper parameters with the best performance
- OnevsAll.ipynb: one vs all classification using logistic regression
- DataExploration.ipynb: loading and reformatting data to have binary drug consumption values
- RB_RH_HL_finalreport.pdf: the analyses written up with evaluation metrics 



