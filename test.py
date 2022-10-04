with open("./roi.txt","r") as f:
    roi = {}
    roi_idx = 0
    for line in f.readlines():
        h1, h2, w1, w2, _ =  line.split(" ")
        roi[str(roi_idx)] = {"h1":h1,"h2":h2,"w1":w1,"w2":w2} 
        roi_idx += 1

for num,dic in roi.items():
    print(dic)