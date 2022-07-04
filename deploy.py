import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

import time
from PIL import Image


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model.h5")



st.title('Vo Nhu Mai graduation thesis')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Vo Nhu Mai graduation thesis')
st.sidebar.subheader('App modes')

# @st.cache()
label = "WARM UP"
n_time_steps = 40
lm_list = []
Action = [  "STOP",
            "THIS MARSHALLER",
            "PROCEED TO NEXT MARSHALLER ON THE RIGHT",
            "PROCEED TO NEXT MARSHALLER ON THE LEFT",
            "PERSONNEL APPROACH AIRCRAFT ON THE RIGHT",
            "PERSONNEL APPROACH AIRCRAFT ON THE LEFT",
            "NORMAL",
            "TURN TO THE LEFT",
            "TURN TO THE RIGHT",
            "SLOW DOWN",
            "MOVE FORWARD"
        ]


def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        #print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

#Đưa vào nhận diện
def detectpose(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    #print(lm_list.shape)
    results = model.predict(lm_list)
    xacsuat = results.tolist()
    xacsuat = xacsuat[0]
    hanhdong = xacsuat.index(max(xacsuat))
    label = Action[hanhdong]
    print(label)
    return label

app_mode = st.sidebar.selectbox('Please Select',
                                ['About My Project','Detect signal']
                                )

if app_mode == 'About My Project':
    st.markdown(
        'In this application we are using **MediaPipe** for creating a Pose Track Points, LSTM to detect signal. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.video('https://www.youtube.com/watch?v=5KCU1mKV1Kk')

    st.markdown('''
          # About Me \n 
            I am ** Vo Nhu Mai ** from class **17ĐHKT01**. \n

            This is my graduation thesis \n

           This application will check the movement of aircraft ramp marshaller. There are 2 modes: Detect Signals and Check Signal. \n

            So choose the mode you want and have fun!!

            ''')
elif app_mode == 'Detect signal':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()

    kpi1, kpi2= st.beta_columns(2)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Signal**")
        kpi2_text = st.markdown("0")


    st.markdown("<hr/>", unsafe_allow_html=True)

    #label = detectpose(model, lm_list)

    cap = cv2.VideoCapture(0)

    i = 0
    warmup_frames = 60

    # mpPose = mp.solutions.pose
    # pose = mpPose.Pose()
    # mpDraw = mp.solutions.drawing_utils

    with pose:

        while True:

            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            i = i + 1
            if i > warmup_frames:
                if results.pose_landmarks:
                    c_lm = make_landmark_timestep(results)
                    lm_list.append(c_lm)
                    if len(lm_list) == n_time_steps:
                        # Nhận diện
                        t1 = threading.Thread(target=detectpose, args=(model, lm_list,))
                        t1.start()
                        lm_list = []

                    img = draw_landmark_on_image(mpDraw, results, img)

            #img = draw_class_on_image(label, img)

            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{label}</h1>", unsafe_allow_html=True)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) == ord('q'):
                break

    #
    # while cap.isOpened():
    #     success, img = cap.read()
    #     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     results = pose.process(imgRGB)
    #
    #     if results.pose_landmarks:
    #         c_lm = make_landmark_timestep(results)
    #         lm_list.append(c_lm)
    #         if len(lm_list) == n_time_steps:
    #             # Nhận diện
    #             t1 = threading.Thread(target=detectpose, args=(model, lm_list,))
    #             t1.start()
    #             lm_list = []
    #
    #             a = detectpose(model, lm_list)
    #             print('aaaaaaaaaa', a)
    #             #kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{a}</h1>", unsafe_allow_html=True)
    #
    #         img = draw_landmark_on_image(mpDraw, results, img)
    #     #img = draw_class_on_image(label, img)
    #     cv2.imshow("Image", img)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    #
    #
    #     # Dashboard
    #     #kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
    #     kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{a}</h1>", unsafe_allow_html=True)

cap.release()

