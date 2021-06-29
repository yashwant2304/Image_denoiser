# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:05:28 2021
@author: Yashwant Bhaidkar
"""

import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
#from tensorflow.keras.layers import Dropout
#from keras.layers.advanced_activations import PReLU
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
from flask import Flask, render_template,flash,request,redirect,url_for
import custom_objects as co #loading custom objects
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('ind.html')

@app.route('/',methods = ['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('no file')
        return redirect(request.url)
    file = request.files['file']
    print('file')
    if file.filename == '':
        flash('no image selected')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        print('reached')
        img = cv2.imread('static/uploads/'+filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #when we load image using cv2 it is in BGR format, so converting it ot standard form RGB
        img = cv2.resize(img, (256, 256))
        input_image = np.expand_dims(img, axis=0)
        model = tf.keras.models.load_model('parallelUnet_model.hdf5',\
                             custom_objects={'dilated_UNET':co.dilated_UNET,'Traditional_UNET':co.Traditional_UNET})
        predicted_image = model.predict(input_image)
        pred = predicted_image[0]
        pred/=255
        img = cv2.convertScaleAbs(pred, alpha=(255.0))
        name = 'static/uploads/free'+filename
        print(name)
        #plt.imsave(name,img)
        cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #imgn = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        #print(imgn)
        flash('success')
        print(filename)
        name2 = 'free'+filename
        print(name2)
        return render_template('ind.html',filename = filename,filename2 = name2)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display/<filename>')
def display_image2(filename2):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename2='uploads/' + filename2), code=302)
        

if __name__ == "__main__":
    app.run()