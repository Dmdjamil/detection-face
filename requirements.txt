import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Webcam test")

webrtc_streamer(key="example")
