import streamlit as st
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
from PIL import Image # Strreamlit works with PIL library very easily for Images
import cv2
model = keras.models.load_model('/content/drive/MyDrive/Python/Major Project/horse_human2.hdf5')
from keras.applications.mobilenet_v2 import preprocess_input,decode_predictions
st.title("Image - CLassifier")
upload = st.file_uploader('Label=Upload the image')
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  x = cv2.resize(opencv_image,(224,224))
  x =x.reshape((224,224,3))
  x = image.img_to_array(x)
  x = np.expand_dims(x,axis=0)
  y = model.predict(x)
  y = np.array(y)
  a=int(y[0][0])
  categories = ['Human','Horse']
  st.title(categories[a])
