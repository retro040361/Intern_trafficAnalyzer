#!/home/retr0ouo/anaconda3/bin/python
from fastapi import FastAPI
from pydantic import BaseModel
import os
import subprocess

app = FastAPI()

class req(BaseModel):
    port: str = None
    arr: list

class cfg(BaseModel):
    left:str = "enable"
    right:str = "enable"
    stop:str = "enable"
    straight:str = "enable"

class yolo(BaseModel):
    batch: str = "16"
    epoch: str = "50"
    trainPath: str = None
    validPath: str = None
    nc : str = None
    names : list = None
    weightPath : str = None
    gpu : bool = False
class classifier(BaseModel):
    inputPath:str = None
    savePath : str = None
    model_name : str = None
    train_size : float = 0.85
    test_size : float = 0.15
    batchSize : int = 16
    eps : int = 100
    lr : float = 0.000003
    weight_decay : float = 0.0001

@app.post("/")
def root():
    return {"message": "Hello World"}

@app.post("/tracker/")
def run_tracker(input: req):
    subprocess.run(["/home/retr0ouo/anaconda3/bin/streamlit","run","./frontend.py"])#,stdout = subprocess.PIPE)
    return {"127.0.0.1":input.port}# ,"output":out.stdout}

@app.post("/test/")
def test(input:req):
    print("I want to go to sleeppppppppppppppppppppp!!!")
    
@app.post("/config/")
def changeConfig(input:cfg):
    with open("/home/retr0ouo/0718test/Yolov5_DeepSort_Pytorch/config.txt","w") as f:
        f.write("turn_left "+cfg.left+"\n")
        f.write("turn_right "+cfg.right+"\n")
        f.write("stop "+cfg.stop+"\n")
        f.write("straight "+cfg.straight+"\n")
    return {"Change":"Done"}

@app.post("/train/illegal")
def getIllgeal():
    return

@app.post("/train/yolo/")
def trainYolo(hyperP:yolo):
    yoloPath = "/home/retr0ouo/0718test/Yolov5_DeepSort_Pytorch/yolov5"
    batchN = hyperP.batch   # batch size
    epochN = hyperP.epoch   # epoch num
    # yaml = hyperP.yaml  # yaml path
    weight = hyperP.weightPath  # weight path
    gpu = hyperP.gpu    # whether using gpu (boolean)

    with open("./data.yaml","w") as f:
        f.write("train: "+hyperP.trainPath+"\n")
        f.write("val: "+hyperP.validPath+"\n")
        f.write("nc: "+hyperP.nc+"\n")
        f.write("names: "+ str(hyperP.names)+"\n")
    command = [f"python3",f"{yoloPath}/train.py","--freeze","10","--img","416","--batch",f"{batchN}","--epochs",f"{epochN}","--data","./data.yaml","--weights",f"{weight}"]
    if gpu:
        command += "--device" 
        command += "0"
    subprocess.run(command)
    exps = os.listdir(f"{yoloPath}/runs/train/")
    return {"trained_model":f"{yoloPath}/runs/train/exp{str(len(exps))}/weights/best.pt","train_resultImg":f"{yoloPath}/runs/train/exp{str(len(exps))}/results.png"}


@app.post("/train/classifier/")
def trainClassifier(hyper:classifier):
    with open("./hyper_classifier.txt","w") as f:
        for key,value in hyper:
            if type(value) != str:
                 value = str(value)
            f.write(value+"\n")
    subprocess.run(["python3","./trainClassifier.py"])
    return {"save path":hyper.savePath,"model name":hyper.model_name}