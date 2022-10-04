import requests
test ={
    "inputPath" : "/home/retr0ouo/0718test/data/patternData_100/",
    "savePath"  : "/home/retr0ouo/0718test/tmp/",
    "model_name" : "model.pth",
    "train_size"  : 0.85,
    "test_size" : 0.15,
    "batchSize" : 16,
    "eps" : 1,
    "lr" : 0.000003,
    "weight_decay" : 0.0001
}

# for i in test:

# f = open("./1231.txt","r")
# tmp = f.readlines()
# for i in tmp:
#     print(i[:-1])
#     print(i)
#     print("---------")

# f.close()

# res = requests.post("http://127.0.0.1:8000/train/classifier/",json=test)
# print(res.json()) 

with open("/home/retr0ouo/0718test/Yolov5_DeepSort_Pytorch/config.txt","r") as f:
    lines = f.readlines()
    sz = len(lines)
    for i in range(sz):
        if(lines[i].split(" ")[1][:-1] == "disable"):
            print(lines[i].split(" ")[0])