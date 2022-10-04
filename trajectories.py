
import pandas as pd
import os
import cv2
import subprocess
import numpy as np
from colour import Color

def save_video(mainPath,videoPath,resultPath,uploaded_video):
    if not os.path.isdir(videoPath):
        os.mkdir(videoPath)
    if not os.path.isdir(resultPath):
        os.mkdir(resultPath)    
    vid = uploaded_video.name
    fullPath = videoPath + vid
    with open(fullPath, mode='wb') as f:
        f.write(uploaded_video.read())  # save video to disk

def runDeepSort(mainPath,videoPath,resultPath,uploaded_video):
    # if not os.path.isdir(videoPath):
    #     os.mkdir(videoPath)
    # if not os.path.isdir(resultPath):
    #     os.mkdir(resultPath)    
    # vid = uploaded_video.name
    # fullPath = videoPath + vid
    # with open(fullPath, mode='wb') as f:
    #     f.write(uploaded_video.read())  # save video to disk

    files = [item for item in os.listdir(videoPath) if (item.endswith(".mp4") or item.endswith(".mov"))]
    string_1 = "python3 " + mainPath + "track.py --source " + videoPath
    string_2 = " --yolo-weights " + mainPath + "/yolov5m.pt --class 2 3 5 7 --project " + \
        resultPath+" --save-vid --save-txt --save-crop --conf-thres 0.5 --iou-thres 0.3\n"

    command = string_1+str(files[0])+string_2
    os.system(command)


def randomColor():
    color = tuple(np.random.random(size=3) * 256)
    return color

def draw_line(id, img, log_file):
    objectFile = log_file.loc[log_file['id'] == id]
    if objectFile.shape[0] == 0:
        # the object didn't been detected
        return img, None, None,None
    if objectFile.shape[0] <= 15:
        # filter for those object whose trajectories is like a point
        return img, None, None, None

    point = []
    # color = randomColor()
    color = (255, 255, 255)  # white
    log = []

    ## variable for color gradient and cropped img
    flag = False
    badTraj = False   # recording the time that a bad trajectory happend
    red = Color("red")
    colors = list(red.range_to(Color("green"),objectFile.shape[0]))
    preVec = None   # use this to record previous vector to calculate the angle
  
    left = objectFile.iloc[:,2]
    top = objectFile.iloc[:,3]
    height = objectFile.iloc[:,5]
    bottom = top + height
    
    max_x = left.max()#.item()
    min_x = left.min()#.item()

    max_y = bottom.max()#.item()
    min_y = bottom.min()#.item()

    w = img.shape[1]
    h = img.shape[0]

    for i in range(objectFile.shape[0]):
        left = objectFile.iloc[i, 2]
        top = objectFile.iloc[i, 3]
        width = objectFile.iloc[i, 4]
        height = objectFile.iloc[i, 5]
        clsName = objectFile.iloc[i, 6]
        frame = objectFile.iloc[i, 0]
        x = int(left + width/2)
        y = int(top + height)
        
        if x > w-50 or x < 50 or y > h-50 or y < 50:
            if i > int(objectFile.shape[0]/2):
                # car drive out of the screen
                break
        point.append([x, y])
        r,g,b = (colors[i].get_rgb())
        if i == 0:
            img = cv2.arrowedLine(img, (x, y), (x, y),
                                  tuple((r*255,g*255,b*255)), 10, tipLength=0.5)
        else:
            img = cv2.arrowedLine(
                img, (point[i-1][0], point[i-1][1]), (x, y), tuple((r*255,g*255,b*255)), 10, tipLength=0.5)
        dataStr = str((x, y, left, top, width, height, clsName))
        log.append([frame, dataStr])
    
    ## range of cropped img
    if min_x > 100:
        min_x -= 100
    else:
        min_x = 0

    if max_x < w-100:
        max_x += 100
    else:
        max_x = w
    if min_y > 100:
        min_y -= 100
    else:
        min_y = 0
    
    if max_y < h-100:
        max_y += 100
    else:
        max_y = h
    # print(f"min_y:{min_y}, max_y: {max_y}, min_x:{min_x}, max_x: {max_x}")
    crop_img = img[min_y:max_y, min_x:max_x]

    avg_pt = np.mean(point,axis=0)
    total_frame = objectFile.shape[0]
    state_string =  f"{avg_pt[0]} {avg_pt[1]} {total_frame}"

    return crop_img, log, clsName, state_string

def Tracker(obj_path, save_path, exp_name,width,height):
    log_file = pd.read_csv(obj_path, sep=" ", header=None)
    log_file = log_file.iloc[:, [0, 1, 2, 3, 4, 5, 10]]
    start_frame = log_file.iloc[0, 0]
    total_frame = log_file.iloc[log_file.shape[0]-1, 0]
    log_file.columns = ['frame', 'id', 'left',
                        'top', 'width', 'height', 'class']
    log_file.sort_values(by=['id', 'frame'])
    os.system(f"mkdir {save_path}/image")
    os.system(f"mkdir {save_path}/avg_pt")
    total_id = log_file.iloc[log_file.shape[0]-1, 1]
    count = 1
    total_log = []
    empty = []
    frame = [i for i in range(total_frame+1)]
    frame = pd.DataFrame(frame)
    frame.columns = ['frame']
    for id in range(1, total_id+1):
        img = np.zeros((height, width, 3), np.uint8)
        img, log, clsName, state_str = draw_line(id, img, log_file)
        if log != None:
            df = pd.DataFrame(log)
            df.columns = ["frame", id]
            df = df.set_index('frame')
            total_log.append(df)
        else:
            empty.append(id)
            continue
        cv2.imwrite(f'{save_path}/image/{exp_name}_{clsName}_{id}.jpg', img)
        with open(f'{save_path}/avg_pt/{clsName}_{id}.txt','w') as f:
            f.write(state_str)
    record = pd.concat(total_log, axis=1)
    record = pd.concat([frame, record], axis=1)
    record.to_csv(f'{save_path}/record.csv', index=False)