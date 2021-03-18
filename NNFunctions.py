import numpy as np
import pandas as pd
import sklearn.metrics as sklm
from sklearn.model_selection import KFold


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms

ind, outd = 12, 5

## Splits data in to training and test sets
## Input a np array of data and the proportion(btwn 0 and 1) in the test set
## Default proportion is 0.25
## Outputs two np arrays with lengths defined by proportion
def train_testSplit(data,prop = 0.25):
    n, p = data.shape
    testn = int(n * prop)
    trainN = n - testn

    # indices for test set
    indtest = np.random.choice(n, testn, replace=False)

    # initialize final np arrays to return
    traindata = np.zeros((trainN, p))
    testdata = np.zeros((testn, p))

    testi = 0
    traini = 0
    for i in range(n):
        if i in indtest:
            testdata[testi] = data[i, :]
            testi += 1
        else:
            traindata[traini] = data[i, :]
            traini += 1

        if traini > trainN:
            ## shouldnt happen
            print("error")
    return traindata, testdata

## create a pytorch dataset object. Pass in data
class drugData(Dataset):
    def __init__(self, data):
        self.n = data.shape[0]
        self.x = torch.from_numpy(data[:, 1:13])
        # print(self.x.dtype)
        self.y = torch.from_numpy(data[:, 13:])

        self.x, self.y = self.x.type(torch.DoubleTensor), self.y.type(torch.DoubleTensor)

        # print(self.x.dtype)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.n


# Three layer neural network class
# Can pass hidden layer size as an arguement
class Model(nn.Module):

    def __init__(self, hlayer):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(ind, hlayer)
        self.lin2 = nn.Linear(hlayer, outd)

    def forward(self, x):
        a1 = torch.relu(self.lin1(x))
        a2 = torch.sigmoid(self.lin2(a1))
        return a2


# trains a neural network
# input: model, a trainloader with data, optimizing funtion, criterion/loss function, epochs, whether to print, what interval to print at
# returns a list tracking loss per epoch
def train(model, criterion, optimizer,trainloader, epochs = 50,p = True, pinterval = 1):
    lossList = []
    for i in range(epochs):
        runningLoss = 0
        for x, y in trainloader:
            optimizer.zero_grad()
            yhat = model(x.float())
            loss = criterion(yhat.float(), y.float())
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
        if p == True:
            if i % pinterval == 0:
                print('epoch ', i, ' loss: ', str(runningLoss))
        lossList.append(runningLoss)
    return lossList

## get predictions from a trained network
## Input: trained model, data, whether to normalize outputs, whether to round to 0/1
## Returns an np array of predictions made by the network passed in
def getPredictions(model,data,normalize = False, roundtoint = False):
    yhat = []
    for i in range(len(data)):
        with torch.no_grad():
            yhat1 = model(data[i][0].float()).numpy()
            if normalize:
                yhat1 = yhat1/sum(yhat1)
            yhat.append(yhat1)
    if roundtoint:
        yhat = np.rint(yhat)
    yhat = np.array(yhat)
    return yhat


## Performs cross validation with the data and parameters passed in
## Returns hamming, subset accuracy, and weighted accurace averaged over the number of folds
def get_cvError(traindata,lr,hsize,folds = 5):
    kfolds = folds
    kf = KFold(n_splits=5)
    kf.get_n_splits(traindata)
    hamming,subsetacc, aucWeighted = 0, 0, 0
    for train_ind, test_ind in kf.split(traindata):
        #print(len(test_ind))
        traint, test = traindata[train_ind], traindata[test_ind]
        trainDataset = drugData(traint)
        testDataset = drugData(test)
        trainloader = torch.utils.data.DataLoader(trainDataset,
                                              batch_size=32,
                                              shuffle=True)

        drugnet = Model(hsize)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(drugnet.parameters(), lr=lr)
        results = train(drugnet,criterion,optimizer,trainloader,200,pinterval=50, p=False)

        yhatTest = getPredictions(drugnet, testDataset,normalize=False,roundtoint=True)
        yTest = test[:,13:]

        hamming += sklm.hamming_loss(yTest,yhatTest)
        subsetacc += sklm.accuracy_score(yTest,yhatTest)
        yhatTest = getPredictions(drugnet, testDataset,normalize=False,roundtoint=False)
        aucWeighted += sklm.roc_auc_score(yTest,yhatTest,average='weighted')
    hamming = hamming/kfolds
    subsetacc = subsetacc/kfolds
    aucWeighted = aucWeighted/kfolds
    return(hamming,subsetacc,aucWeighted)

