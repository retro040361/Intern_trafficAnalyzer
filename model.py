
import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.utils.data as data

import torch
import torchvision
import torchvision.models as models

import os
from PIL import Image

from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



class testDataset(Dataset):
    def __init__(self,img_arr,file_list=None,transform=None):
        self.to_tensor = transforms.ToTensor()
        self.image = img_arr
        self.file_list = file_list

        self.data_len = len(self.image)
        self.transform = transform
    def __getitem__(self, index):
        img_as_tensor = None
        if self.transform:
            img_as_tensor = self.transform(self.image[index])
        else:
            img_as_tensor = self.to_tensor(self.image[index])

        file_name = self.file_list[index]
        return (img_as_tensor,file_name)
    def __len__(self):
        return self.data_len

def inference(mainPath,resultPath,vid):
    configPath = mainPath + "/config.txt"
    disable = [False,False,False,False]     # turn_left turn_right stop straight
    with open(configPath,"r") as f:
        lines = f.readlines()
        sz = len(lines)
        for i in range(sz):
            if(lines[i].split(" ")[1][:-1] == "disable"):
                disable[i] = True
    
    use_gpu = torch.cuda.is_available()
    net = models.resnet18(pretrained=False)
    fc_features = net.fc.in_features
    net.fc = nn.Linear(fc_features,4)
    net.load_state_dict(torch.load(mainPath + 'patternModel_0810_gradCrop.pth', map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))

    imgPath = resultPath + "exp/image/"
    file_list = os.listdir(imgPath)

    test_data = []
    for img in file_list:
        test_data.append(np.asarray(Image.open(imgPath+img)))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    testTrans = transforms.Compose([transforms.ToPILImage(), \
                                    transforms.Resize((224,224)), \
                                    transforms.ToTensor(), \
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 
    test_dataset = testDataset(test_data,file_list,transform=testTrans)
    testloader = DataLoader(test_dataset, batch_size=1,shuffle=False, num_workers=0)

    cls_name = {2:"car",3:"motorcycle",5:"bus",7:"truck"}
    classes = ['Turn left', 'Turn right', 'Stop', 'Go Straight']

    objectInfo = []
    with torch.no_grad():
        dataiter = iter(testloader)
        while True:
            try:
                images, filename = dataiter.next()
                prediction = net(images)
                predicted_class = prediction.argmax()
                token = filename[0].split("_")
                clsIdx = token[1]
                id = int(token[2].split(".")[0])
                name = cls_name[int(clsIdx)]
                illegal = 0; 
                if disable[predicted_class]:
                    illegal =  1
                obj = [clsIdx,id,predicted_class.item(), illegal]    # [ class index, object id, prediction, illegal or not]
                objectInfo.append(obj)

                
            except StopIteration:
                print("Classifying is finished")
                break
    objectInfo_df = pd.DataFrame(objectInfo,columns=["class","obj_id","pred","illegal"])
    return objectInfo_df
