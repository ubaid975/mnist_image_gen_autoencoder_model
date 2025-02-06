import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import PIL
import cv2

model=load_model('mnist_autoencoder.h5',compile=False)

def app(image):
    image=cv2.resize(image,(28,28))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imag_1=image.reshape(1,28,28)
    pred=model.predict(imag_1).reshape(28,28)
    pred=cv2.resize(pred,(224,244))
    return PIL.Image.fromarray(pred)
app=gr.Interface(fn=app,inputs=['image'],outputs=['image'])
app.launch(share=True)