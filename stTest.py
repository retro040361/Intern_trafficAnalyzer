import streamlit as st
import streamlit_modal as modal

import streamlit.components.v1 as components


open_modal = st.button("Config")
if open_modal:
    modal.open()

if modal.is_open():
    with modal.container():
        st.header("Define Illegal Motion Pattern.")

        action = ["turn_left","turn_right","stop","straight"]
        state = ["enable","enable","enable","enable"]
        st.write("Click the check box below")
        left = st.checkbox("Turn left")
        if left:
            state[0] = "disable"
        right = st.checkbox("Turn right")
        if right:
            state[1] = "disable"
        stop = st.checkbox("Stop")
        if stop:
            state[2] = "disable"
        straight = st.checkbox("straight")
        if straight:
            state[3] = "disable"

        with open("/home/retr0ouo/0718test/Yolov5_DeepSort_Pytorch/config.txt","w") as f:
            for i in range(4):
                f.write(action[i] + " " +state[i] + "\n")
