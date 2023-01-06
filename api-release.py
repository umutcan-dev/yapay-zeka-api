from flask import Flask, request
from flask_restful import Api, Resource
from flask_ngrok import run_with_ngrok
import pandas as pd
from PIL import Image
import keras
from keras import backend as K
from keras.layers.core import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import img_to_array
from flask import jsonify
import io
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras import applications
from keras import applications
from tensorflow.keras.models import Sequential


gdown --id '1qZ5oy3NwjBqozIcc7JItdFjCqB25FEaM' #h5 download
gdown --id '1YZNf6fFFoHuYEe-wf-93dOhe2kfuEPJE' #.csv file download


app = Flask(__name__)
api = Api(app)

global image

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    print(np.shape(image))
    
    return image


def get_model():
    global model
    global graph
    vgg16_model = applications.vgg16.VGG16()
    model = tf.keras.Sequential()


    for i in vgg16_model.layers:
        model.add(i)


    for layer in model.layers:
        layer.trainable = False


    model.add(Dense(4, activation='softmax'))
    model.make_predict_function('model_weights.h5')
    graph = tf.compat.v1.get_default_graph()
    print("Model loaded!")


print(" Loading Keras model...")
get_model()

import csv
csvFile = 0;

with open('csvornek.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  for lines in csvFile:
		  fileList = lines


class NeYedim(Resource):
      def get(self):
        
        data = {
            'prediction': {
            'get' : 'get işlemi',
            }
        }

        
        print(data)
        return {'data' : data}, 200
      
      #post işlemi
      def post(self):
        name = request.form['name']
        encoded = request.form['image']
        decoded = base64.b64decode(encoded)
        decode = io.BytesIO(decoded)
        image = Image.open(decode)
        image.save("picture.jpg")
        image = Image.open("picture.jpg")

        processed_image = preprocess_image(image, target_size=(224, 224))

        with graph.as_default():
            vgg16_model = applications.vgg16.VGG16()
            model = tf.keras.Sequential()
        
            for i in vgg16_model.layers:
                model.add(i)
        
            for layer in model.layers:
                layer.trainable = False
        
            model.add(Dense(4, activation='softmax'))
            #model.make_predict_function('/content/drive/MyDrive/model_weights.h5')  
            model.load_weights('/content/drive/MyDrive/model_weights.h5')
            prediction = model.predict(processed_image).tolist()


        prediction_max = max(prediction[0])
        print(prediction_max)

        id = -1
	sizeOfDemoList = len(fileList)
        for i in range(sizeOfDemoList):
         if(prediction[0][i]==prediction_max):
           id=i
        print(id)


        data = {
            'prediction': {
            'name' :  name,
            'post' : 'post işlemi',
            'sonuc': fileList[id]
            }
        }
f
        
        print(data)
        return {'data' : data}, 200

# Add URL endpoints
api.add_resource(NeYedim, '/neyedinanaliz')

if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000)
    app.run()

