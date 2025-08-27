import streamlit as st
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import subprocess

packages = str(subprocess.run('pip list', capture_output=True))
st.markdown(packages.replace('\\r\\n', '  \\\n'))

st.title("Image Classification Model: Cats and Dogs")

with st.sidebar:
  st.header('Data Format Requirement:')
  st.caption('upload an image file to run the app')
  with st.expander('Image format'):
    st.markdown('- JPG')
    st.markdown('- JPEG')
    st.markdown('- PNG')

if 'clicked' not in st.session_state:
  st.session_state.clicked = {1:False}

def clicked(button):
  st.session_state.clicked[button] = True

st.button("Let's go classy(fy)!", on_click=clicked, args=[1])

def pre_processing_img(img):
    img = np.asarray(img)
    img = cv2.resize(img, (32,32))
    img = np.expand_dims(img/255, 0)
    return img

def cat_or_dog(pred):
  if(pred < 0.5): 
    return 'Hi Kitty!'
  else:
    return 'Hey Doggo!'

if st.session_state.clicked[1]:
  uploaded_img = st.file_uploader('Browse files', type=['jpg','jpeg' ,'png'])
  
  if uploaded_img:
    st.subheader('Your Pic')
    st.image(uploaded_img)
    
    model = tf.keras.models.load_model(os.path.join('./model','model.h5'))
    img = Image.open(uploaded_img)
    img = pre_processing_img(img)
    pred = model.predict(img)

    st.subheader('My Guess:')
    st.write(f"<p style='text-align:center; font-size:3rem; font-weight:bold; text-shawdow:2px 2px 3px blue;  color:{'#d6213b' if 'kitty' else '#9fdcf7'}'>{cat_or_dog(pred)}</p>", unsafe_allow_html=True)
    st.write(pred)


