import torch
import torch.nn as nn
import torchvision
from torchvision import datasets ,models,transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.utils.data as data
import shutil
from torch.utils.data import random_split
import torchvision.models as models
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score,accuracy_score
import pandas as pd

f = open("./hyper_classifier.txt","r")
para = f.readlines()

# parameter
input_path = para[0][:-1]
savePath = para[1][:-1]
model_name = para[2][:-1]
train_size = float(para[3][:-1])
test_size = float(para[4][:-1])
batchSize = int(para[5][:-1])
eps = int(para[6][:-1])
lr = float(para[7][:-1])
weight_decay = float(para[8][:-1])


def dataLoader(inputPath,train_size,test_size,batchSize):
    # data transformation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    trainTrans = transforms.Compose([transforms.Resize((224,224)),\
                      transforms.RandomRotation(30, expand=True),\
                      transforms.RandomAffine(degrees=(10, 30),translate=(0.25, 0.5), scale=(0.5,0.9)),\
                      transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    testTrans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 

    # data loader
    data_set = datasets.ImageFolder(inputPath)
    dataSize = len(data_set)
    train_set,  test_set = random_split(data_set,[int(dataSize*train_size),dataSize - int(dataSize*train_size)])

    train_set.dataset.transform = trainTrans
    test_set.dataset.transform = testTrans

    trainLoader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=0) 
    testLoader = torch.utils.data.DataLoader(test_set, batch_size=batchSize, shuffle=False, num_workers=0) 
    classes = data_set.classes
    return trainLoader, testLoader, classes

def train(trainLoader,testLoader,eps,lr,weight_decay,classes,savePath,model_name):
    #log = open(savePath+"log.txt","w")
    # wheter using gpu
    use_gpu = torch.cuda.is_available()

    # load model
    net = models.resnet18(pretrained=False)
    fc_features = net.fc.in_features
    features = len(classes)
    net.fc = nn.Linear(fc_features,features)   # 4 patterns including straight、turn right、turn left、stop

    optimizer = torch.optim.Adam(net.parameters(),lr = lr,weight_decay=weight_decay)   # optimize all cnn parameters
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        criterion = criterion.cuda()
        net = net.cuda()

    # Training
    train_pred = []
    train_label = []

    test_pred = []
    test_label = []

    test_loss_min = np.Inf

    total_train_loss = []
    total_test_loss = []
    print("Start training")
    for epoch in range(eps):  
        train_loss = 0.
        test_loss = 0.

        net.train()
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            if use_gpu:
              inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 4 == 0: 
                print(f'[{epoch+1}, {i+1}] loss: {loss.item()}' )
                #f.write(f'[{epoch+1}, {i+1}] loss: {loss.item()}\n' )

        net.eval()
        train_correct = 0.
        train_total = 0.
        test_correct = 0.
        test_total = 0.
        with torch.no_grad():
            for i, data in enumerate(trainLoader, 0):
                inputs, labels = data
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                out = net(inputs)
                loss = criterion(out, labels)
                pred = out.data.max(1, keepdim=True)[1]
                train_loss += loss.item()
                train_pred += pred.cpu().numpy().tolist()
                train_label += labels.cpu().numpy().tolist()
                train_correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
                train_total += inputs.size()[0]
            total_train_loss.append(train_loss)
            #f.write(f'Train Accuracy: {100. * train_correct / train_total}% ({train_correct}/{train_total})\n')
            print(f'Train Accuracy: {100. * train_correct / train_total}% ({train_correct}/{train_total})')
            for i, data in enumerate(testLoader, 0):
                inputs, labels = data
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda() 
                out = net(inputs)
                loss = criterion(out, labels)
                pred = out.data.max(1, keepdim=True)[1]
                test_loss += loss.item()

                test_pred += pred.cpu().numpy().tolist()
                test_label += labels.cpu().numpy().tolist()
                test_correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
                test_total += inputs.size()[0]
            total_test_loss.append(test_loss)
            #f.write(f'Test Accuracy: {100. * test_correct / test_total}% ({test_correct}/{test_total})\n')
            print(f'Test Accuracy: {100. * test_correct / test_total}% ({test_correct}/{test_total})')
            if test_loss <= test_loss_min:
                #f.write(f'Test loss decreased ({test_loss_min} --> {test_loss}).  Saving model ...')
                print(f'Test loss decreased ({test_loss_min} --> {test_loss}).  Saving model ...')
                torch.save(net.state_dict(), savePath+model_name)
                test_loss_min = test_loss
    #f.write('Finished Training')
    print('Finished Training')
    plt.plot(total_train_loss)
    plt.savefig(savePath+"train_loss.jpg")
    plt.plot(total_test_loss)
    plt.savefig(savePath+"test_loss.jpg")
    #f.close()
    score = modelEval(train_label,train_pred,test_label,test_pred,classes)
    return score

def modelEval(train_label,train_pred,test_label,test_pred,classes):
    # Evaluation
    train_cf_mat = confusion_matrix(train_label,train_pred)
    test_cf_mat = confusion_matrix(test_label,test_pred)
    train_df = pd.DataFrame(train_cf_mat,index=classes,columns = classes)
    test_df = pd.DataFrame(test_cf_mat,index=classes,columns = classes)
    
    score = []
    score.append(recall_score(train_label, train_pred,average="macro"))    # Recall
    score.append(precision_score(train_label, train_pred,average="macro")) # Precision
    score.append(f1_score(train_label, train_pred,average="macro"))        # F1 Score
    score.append(accuracy_score(train_label,train_pred))                   # Accuracy

    score.append(recall_score(test_label, test_pred,average="macro"))      # Recall
    score.append(precision_score(test_label, test_pred,average="macro"))   # Precision
    score.append(f1_score(test_label, test_pred,average="macro"))          # F1 Score
    score.append(accuracy_score(test_label,test_pred))                     # Accuracy
    return score

trainLoader, testLoader, classes = dataLoader(input_path,train_size,test_size,batchSize)
score = train(trainLoader,testLoader,eps,lr,weight_decay,classes,savePath,model_name)
print(score)