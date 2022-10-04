
import cv2
import numpy as np


import cv2

def get_images_from_video(video_name):
    video_frame = None
    vc = cv2.VideoCapture(video_name)
    c = 1
    
    if vc.isOpened(): 
        rval, video_frame = vc.read()
    else:
        rval = False

    while rval:   
        rval, video_frame = vc.read()
        
        if(c % 30 == 0): #每隔幾幀進行擷取
            vc.release()
            return video_frame
        c = c + 1
        
    return video_frame


if __name__ == '__main__' :
    video_name = '/Users/zhangchenhao/Desktop/Intern/demoVideo.mp4'    
    img = get_images_from_video(video_name)
    # img = cv2.imread("/Users/zhangchenhao/Desktop/Retro/pxl_good.jpeg")
    
    rs = cv2.selectROIs("window",img)
    with open("./roi.txt","w") as f:
        for r in rs:
            print(f"region {int(r[1])}:{int(r[1]+r[3])}, {int(r[0])}:{int(r[0]+r[2])}")
            h1,h2,w1,w2 = int(r[1]),int(r[1]+r[3]), int(r[0]),int(r[0]+r[2])
            f.write(f"{h1} {h2} {w1} {w2} \n")
    # imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] # 0-1080  , 0:1920
    # gray = cv2.cvtColor(imCrop,cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    # cont, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # with open("./roi.txt","w") as f:
    #    h1,h2,w1,w2 = int(r[1]),int(r[1]+r[3]), int(r[0]),int(r[0]+r[2])
    #     # print(f"region {int(r[1])}:{int(r[1]+r[3])}, {int(r[0])}:{int(r[0]+r[2])}")
    #    f.write(f"{h1} {h2} {w1} {w2}")
    # # print(img.shape)
    # cv2.imshow("Image", imCrop)
    # cv2.waitKey(0)