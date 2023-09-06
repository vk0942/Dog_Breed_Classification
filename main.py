from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask import request
from flask import session
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
#This
from numpy import expand_dims
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
 #This
import numpy as np  
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt  
import math  
import cv2
top_model_weights_path_inc = r'bottleneck_modelinceptionv3merge.h5'
top_model_weights_path_vgg = r'bottleneck_modelVGG16merge.h5'
#useless
import pickle
with open(r'classindices.pkl', 'rb') as f:
    class_dictionary = pickle.load(f)
# model.add(Dense(num_classes, activation='sigmoid'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()  
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        img_rel_path=os.path.join("files/",secure_filename(file.filename))
        image_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        session["image"]=image_path
        session["img_rel"]=img_rel_path
        return render_template("choose.html",img=image_path)
   
 # get the prediction label  
        # print("Image ID: {}, Label: {}".format(inID, label)) 
        # return "The given dog is a {}".format( label)
    return render_template('index.html', form=form)
@app.route('/inc', methods=['GET',"POST"])
def inc():
    image_path =session.get("image")
    rel_path=session.get("img_rel")
    orig = cv2.imread(image_path)  
    image = load_img(image_path, target_size=(224, 224))  
    image = img_to_array(image)  
    print("[INFO] loading and preprocessing image...")  
    # image = load_img(file, target_size=(224, 224))  
    image = img_to_array(image)  

    image = image / 255
    
    image = np.expand_dims(image, axis=0)
    model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')  
   
 # get the bottleneck prediction from the pre-trained VGG16 model  
    bottleneck_prediction = model.predict(image) 
    #inceptionv3
    model = Sequential()  
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))  
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.005))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.05))
    # model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.01))
    # model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.005))
    # model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.005))
    # model.add(BatchNormalization())
    # model.add(BatchNormalization())
    model.add(Dense(70, activation='softmax'))
    model.load_weights(top_model_weights_path_inc,by_name=True)  
    predict_x = model.predict(bottleneck_prediction)
    classes_predicted=np.argmax(predict_x,axis=1)
    inID = classes_predicted[0]  

    # class_dictionary = generator_top.class_indices

    inv_map = {v: k for k, v in class_dictionary.items()}  

    label = inv_map[inID]
    return render_template("result.html",img=rel_path,txt="Given dog is a {}".format(label))

@app.route('/vgg16', methods=['GET',"POST"])
def vgg16():
    image_path =session.get("image")
    rel_path=session.get("img_rel")
    orig = cv2.imread(image_path)  
    image = load_img(image_path, target_size=(224, 224))  
    image = img_to_array(image)  
    print("[INFO] loading and preprocessing image...")  
    # image = load_img(file, target_size=(224, 224))  
    image = img_to_array(image)  

    image = image / 255
    
    image = np.expand_dims(image, axis=0)
    model = applications.VGG16(include_top=False, weights='imagenet')  
   
    bottleneck_prediction = model.predict(image) 
    #inceptionv3
    model = Sequential()  
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))  
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.005))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.05))
    # model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.05))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.05))
    # model.add(BatchNormalization())
    model.add(Dense(70, activation='softmax'))
    model.load_weights(top_model_weights_path_vgg,by_name=True)  
    predict_x = model.predict(bottleneck_prediction)
    classes_predicted=np.argmax(predict_x,axis=1)
    inID = classes_predicted[0]  

    # class_dictionary = generator_top.class_indices

    inv_map = {v: k for k, v in class_dictionary.items()}  

    label = inv_map[inID]
    return render_template("result.html",img=rel_path,txt="Given dog is a {}".format(label))


if __name__ == '__main__':
    app.run(debug=True)