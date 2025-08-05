import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from helper import preprocess_frame, emotion_labels

model = load_model('expression_model.h5')

st.title("Real-Time Facial Expression Recognition")
start = st.button("Start Camera")

if start:
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = preprocess_frame(frame)

        for face, (x, y, w, h) in results:
            prediction = model.predict(face)
            label = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
