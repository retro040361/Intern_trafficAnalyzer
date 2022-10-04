
import streamlit as st
import pandas as pd
import os
import cv2
import subprocess
import numpy as np
import os
from PIL import Image
from model import inference
from trajectories import runDeepSort, Tracker, save_video

import streamlit_modal as modal
import streamlit.components.v1 as components

if 'run' not in st.session_state:
	st.session_state.run = True
if 'vidName' not in st.session_state:
	st.session_state.vidName = None

def illPark(df,resultPath,roi):
    print(roi)
    rows = df.shape[0]
    df["illPark"] = 0
    # print(df)
    # print(df.shape)
    for i in range(rows):
        print(f"i:{i}")
        print(df.iloc[i,:])
        cls_id = df.iloc[i,0]
        # print(cls_id)
        obj_id = str(df.iloc[i,1])
        # print(obj_id)
        w,h,frame = 0,0,0
        with open(resultPath+f"exp/avg_pt/{str(cls_id)}_{str(obj_id)}.txt","r") as f:
            w,h,frame =  f.readlines()[0].split(" ")
        print(w,h)
        for num, dic in roi.items():
            if w>dic["w1"] and w<dic["w2"] and h > dic["h1"] and h<dic["h2"]:
                print("In")
                df.iloc[i,-1] = 1
    return df     

def iterData(df,resultPath):
    if df.shape[0] == 0:
        st.text("This category has no result, please try other options.")
        return
    
    imgPath = resultPath + "exp/image/"
    cls_name = {2:"car",3:"motorcycle",5:"bus",7:"truck"}
    classes = ['Turn left', 'Turn right', 'Stop', 'Go Straight']
    rows = df.shape[0]
    for i in range(rows):
        cls_id = df.iloc[i,0]
        obj_id = str(df.iloc[i,1])
        pred = df.iloc[i,2]
        illegal = df.iloc[i,3]
        # crop img
        name = cls_name[int(cls_id)]
        cropPath = resultPath+"exp/crops/"+name+"/"+obj_id+"/"
        cropImgs = os.listdir(cropPath)
        crop_img = Image.open(cropPath+cropImgs[int(len(cropImgs)/2)])
        crop_img = crop_img.resize((200,200))
        # trajectories img
        img = Image.open(imgPath+"exp_"+cls_id+"_"+obj_id+".jpg")
        img = img.resize((200,200))
        st.text(f"[{name}] Object {obj_id}: {classes[pred]}")
        if illegal == 1:   
            st.markdown("**Illegal**")
        else:
            st.markdown("Normal")
        st.image([img,crop_img],width=200)

def get_images_from_video(video_name):
    vc = cv2.VideoCapture(video_name)
    c = 1
    video_frame = None
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

def main():
    st.markdown('Made by **RetroOuO**')
    
    
    st.title('Vehicle Trajectory Analyzer')
    col1, col2,_,_ = st.columns([1,1,1,1])
    
    with col1:
        open_modal = st.button('Config')
    with col2:
        exit = st.button('Exit Program')
    
    if exit:    # st.button('Exit'):
        st.write('exit')
        st.stop()
    # open_modal = st.button("Config")
    if open_modal:
        modal.open()
    
    if modal.is_open():
        with modal.container():
            st.header("Define Illegal Motion Pattern.")
            trans = {"disable":True,"enable":False}
            action = []
            state = []
            
            with open("./config.txt","r") as f:
                for line in f.readlines():
                    action.append(line.split(" ")[0])
                    state.append(line.split(" ")[1][:-1])
            
            # state = ["enable","enable","enable","enable"]
                st.write("Click the check box below")
                left = st.checkbox("Turn left",value = trans[state[0]])
                if left:
                    state[0] = "disable"
                else:
                    state[0] = "enable"
                right = st.checkbox("Turn right",value = trans[state[1]])
                if right:
                    state[1] = "disable"
                else:
                    state[1] = "enable"
                stop = st.checkbox("Stop",value = trans[state[2]])
                if stop:
                    state[2] = "disable"
                else:
                    state[2] = "enable"
                straight = st.checkbox("straight",value = trans[state[3]])
                if straight:
                    state[3] = "disable"
                else:
                    state[3] = "enable"
    
            with open("./config.txt","w") as f:
                for i in range(4):
                    f.write(action[i] + " " +state[i] + "\n")
    
    st.header('Please Upload Your Video')
    st.text('Note that your video must be traffic related, or the app won\'t have any effect.')
    
    uploaded_video = st.file_uploader("Choose a video file")
    
    mainPath = "./"
    roi = dict()
    if uploaded_video is not None: 
        path = mainPath + uploaded_video.name
        if not os.path.isdir(path):
            os.mkdir(path)

        videoPath = path +  "/source/"		# sourece
        resultPath = path + "/result/"
        st.subheader("Main Program")
        st.text("Analyzing the video ...")
        vid = uploaded_video.name
        if vid != st.session_state.vidName:
            if st.session_state.vidName != None:
                os.system(f"rm -r {vid}")
            st.session_state.vidName = vid
            st.session_state.run = True
        with open("./roi.txt","r") as f:
            roi = {}
            roi_idx = 0
            for line in f.readlines():
                h1, h2, w1, w2, _ =  line.split(" ")
                roi[str(roi_idx)] = {"h1":h1,"h2":h2,"w1":w1,"w2":w2}            
                roi_idx+=1
            # h1, h2, w1, w2 =  f.readlines()[0].split(" ")
            # roi = {"h1":h1,"h2":h2,"w1":w1,"w2":w2}
        if st.session_state.run:
            print("pass")
            # run deep sort project
            # save_video(mainPath,videoPath,resultPath,uploaded_video)

            # runDeepSort(mainPath,videoPath,resultPath,uploaded_video)
        st.text("Drawing their trajectories ...")
        results = [item for item in os.listdir(resultPath)]
        video_name = videoPath + vid
        video_img = get_images_from_video(video_name)
        width = video_img.shape[1]
        height = video_img.shape[0]
        print(roi)
        for exp in results:
            path = resultPath + str(exp) + "/"
            record = [txt for txt in os.listdir(
                path+"tracks/") if txt.endswith(".txt")]
            if len(record) != 1:
                print("Error file existing")
                break
            txtPath = path + "tracks/" + record[0]
            if st.session_state.run:
                print("Run Tracker")
                Tracker(txtPath, path, exp,width,height)
        st.text("Classifying ...")
        objInfo = inference(mainPath,resultPath,vid)
        st.text("Transforming video...")
        if st.session_state.run:
            print("Transforming video")
            os.system(f"ffmpeg -y -i {resultPath+'exp/'+vid} -vcodec libx264 {resultPath+'exp/video.mp4'}")
        st.subheader("Tracking Result")
        video_file = open(resultPath+'exp/video.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        
        st.session_state.run = False
        actionlist = ["Don't Show result","Show All","Show by Vehicle","Show by Motion Pattern","Show by Illegalness","Show by Illegal Parking"]
        action = st.sidebar.selectbox("Show Result by:",actionlist)
        if action == "Show All":
            sort = ["Ascend","Descend"]
            sortby = st.sidebar.selectbox("Order:",sort)
            if sortby == "Ascend":
                data = objInfo.sort_values("obj_id",ascending = True)
                iterData(data,resultPath)
            if sortby == "Descend":
                data = objInfo.sort_values("obj_id",ascending = False)
                iterData(data,resultPath)
        if action == "Show by Vehicle":
            cls_name = {2:"car",3:"motorcycle",5:"bus",7:"truck"}
            classList = ["car","motorcycle","bus","truck"]
            vehicle = st.sidebar.selectbox("Class:",classList)
            idx = list(cls_name.keys())[list(cls_name.values()).index(vehicle)]
            data = objInfo[objInfo["class"]==str(idx)]
            iterData(data,resultPath)
        if action == "Show by Motion Pattern":
            motionList = ['Turn left', 'Turn right', 'Stop', 'Go Straight']
            motion = st.sidebar.selectbox("Motion Pattern:",motionList)
            idx = motionList.index(motion)
            df = objInfo[objInfo["pred"]==idx]
            iterData(df,resultPath)
        if action == "Show by Illegalness":
            df = objInfo[objInfo["illegal"]==1]
            iterData(df,resultPath)
        if action == "Show by Illegal Parking":
            for num, dic in roi.items():
                video_img = cv2.rectangle(video_img,(int(dic["w1"]),int(dic["h1"])),(int(dic["w2"]),int(dic["h2"])),(0,0,255),cv2.LINE_AA)
            st.image(video_img,width = 400)
            df = objInfo[objInfo["pred"]==0]
            df = illPark(df,resultPath,roi)
            ill_df = df[df["illPark"]==1]
            iterData(ill_df,resultPath)
        #st.stop()


if __name__ == '__main__':
    main()