test ={
    "inputPath" : "/home/retr0ouo/0718test/data/patternData_100",
    "savePath"  : "/home/retr0ouo/0718test/tmp",
    "model_name" : "model.pth",
    "train_size"  : 0.85,
    "test_size" : 0.15,
    "batchSize" : 16,
    "eps" : 1,
    "lr" : 0.000003,
    "weight_decay" : 0.0001
}

for i in test:
    print(test[i])