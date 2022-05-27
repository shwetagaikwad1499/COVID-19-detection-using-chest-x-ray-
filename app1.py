
from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
#from flask_ngrok import run_with_ngrok

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets',
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#run_with_ngrok(app)

@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')


@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   #resnet_chest = load_model('models/RESNET_result.h5')
   #vgg_chest = load_model('models/VGG-16.h5')
   #regularCNN_chest = load_model('models/RegularCNN.h5')

   def create_CNNmodel():
       model1 = tf.keras.models.Sequential([
       tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)),
       tf.keras.layers.MaxPooling2D(2, 2),

       tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
       tf.keras.layers.MaxPooling2D(2, 2),

       tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
       tf.keras.layers.MaxPooling2D(2, 2),

       tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
       tf.keras.layers.MaxPooling2D(2,2),


       tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
       tf.keras.layers.MaxPooling2D(2, 2),

       tf.keras.layers.Dropout(0.2),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(256, activation='relu'),
       tf.keras.layers.Dense(3, activation='softmax')
       ])
       adam = tf.keras.optimizers.Adam(lr=0.0001)
       model1.compile(loss = tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer=adam)

       return model1

   # Create a new model instance
   model1 = create_CNNmodel()

   # Restore the weights
   model1.load_weights('weights/RegularCNN_results.h5')


   def create_RESNETmodel():
        base_model = tf.keras.applications.InceptionResNetV2(
            include_top = False,

            weights = 'imagenet',
            input_shape = (299, 299, 3),
            classifier_activation = 'softmax'

        )
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
        out = tf.keras.layers.Dense(3, activation='softmax', name='dense_output')(x)
        model2 = tf.keras.models.Model(inputs = base_model.input, outputs=out)

        model2.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy']
        )
        return model2



   # Create a new model instance
   model2 = create_RESNETmodel()

   # Restore the weights
   model2.load_weights('weights/RESNET_Results.h5')

   def create_VGGmodel():
       vgg=VGG16(include_top=False,weights='imagenet',
       input_shape=(299, 299, 3))
       for layer in vgg.layers:
           layer.trainable=False
       x=Flatten()(vgg.output)
       prediction_vgg=tf.keras.layers.Dense(3,activation='softmax')(x)
       vgg_model=Model(inputs=vgg.input,outputs=prediction_vgg)
       #plot_model(vgg_model)

       vgg_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
       loss='CategoricalCrossentropy',
       metrics=['accuracy'])
       return vgg_model

   vgg_model = create_VGGmodel()

   # Restore the weights
   vgg_model.load_weights('weights/VGG-16_Results.h5')




   image = cv2.imread('./flask app/assets/images/upload_chest.jpg') # read file
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(299,299))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)

   regularCNN_pred = model1.predict(image)
      #probability = resnet_pred[0]
   #regularCNN_pred.reshape(1,3)
   regularCNN_pred = regularCNN_pred[0]
   print("RegularCNN Predictions:")
   print(regularCNN_pred)
   i =  np.argmax(regularCNN_pred)
   print(i)
   percent_pred = regularCNN_pred[i]
   print(percent_pred)
   if i==0:
       regularCNN_chest_pred=str('%.2f' % (percent_pred*100) + '% COVID')
   elif i==1:
      regularCNN_chest_pred=str('%.2f' %(percent_pred*100) + '% NORMAL')
   else:
      regularCNN_chest_pred=str('%.2f' %(percent_pred*100) + '% VIRAL')
   print(regularCNN_chest_pred)


   resnet_pred = model2.predict(image)
      #probability = resnet_pred[0]
   #regularCNN_pred.reshape(1,3)
   resnet_pred = resnet_pred[0]
   print("Resnet Predictions:")
   print(resnet_pred)
   i =  np.argmax(resnet_pred)
   print(i)
   percent_pred = resnet_pred[i]
   print(percent_pred)
   if i==0:
       resnet_chest_pred=str('%.2f' % (percent_pred*100) + '% COVID')
   elif i==1:
       resnet_chest_pred=str('%.2f' %(percent_pred*100) + '% NORMAL')
   else:
       resnet_chest_pred=str('%.2f' %(percent_pred*100) + '% VIRAL')
   print(resnet_chest_pred)


   vgg_pred = vgg_model.predict(image)
      #probability = resnet_pred[0]
   #regularCNN_pred.reshape(1,3)
   vgg_pred = vgg_pred[0]
   print("VGG16 Predictions:")
   print(vgg_pred)
   i =  np.argmax(vgg_pred)
   print(i)
   percent_pred = vgg_pred[i]
   print(percent_pred)
   if i==0:
       vgg_chest_pred=str('%.2f' % (percent_pred*100) + '% COVID')
   elif i==1:
       vgg_chest_pred=str('%.2f' %(percent_pred*100) + '% NORMAL')
   else:
       vgg_chest_pred=str('%.2f' %(percent_pred*100) + '% VIRAL')
   print(vgg_chest_pred)

   final_pred = (resnet_pred + vgg_pred + percent_pred)/3
   print("Final Predictions:")
   print(final_pred)
   i =  np.argmax(final_pred)
   print(i)
   percent_pred = final_pred[i]
   print(percent_pred)
   if i==0:
       final_chest_pred=str('%.2f' % (percent_pred*100) + '% COVID')
   elif i==1:
       final_chest_pred=str('%.2f' %(percent_pred*100) + '% NORMAL')
   else:
       final_chest_pred=str('%.2f' %(percent_pred*100) + '% VIRAL')
   print(final_chest_pred)



   #resnet_pred = resnet_chest.predict(image)
   #probability = resnet_pred[0]
   #print("Resnet Predictions:")
   #if probability[0] > 0.5:
    #   resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')
   #elif probability[0] < 0.5:
    #   resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% VIRAL')
   #else:
    #   resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   #print(resnet_chest_pred)

   #vgg_pred = vgg_chest.predict(image)
   #probability = vgg_pred[0]
   #print("VGG Predictions:")
   #if probability[0] > 0.5:
    #   vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')
   #elif probability[0] < 0.5:
    #   vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% VIRAL')
   #else:
    #   vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   #print(vgg_chest_pred)


   #regularCNN_pred = regularCNN_chest.predict(image)
   #probability = regularCNN_pred[0]
   #print("CNN Predictions:")
   #if probability[0] > 0.5:
    #   regularCNN_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')
   #elif probability[0] < 0.5:
    #   regularCNN_chest_pred = str('%.2f' % (probability[0]*100) + '% VIRAL')
   #else:
    #   regularCNN_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   #print(regularCNN_chest_pred)





   #return render_template('results_chest.html',resnet_chest_pred=resnet_chest_pred,vgg_chest_pred=vgg_chest_pred,regularCNN_chest_pred=regularCNN_chest_pred)
   #return render_template('results_chest.html',regularCNN_chest_pred=regularCNN_chest_pred,resnet_chest_pred=resnet_chest_pred,vgg_chest_pred=vgg_chest_pred)
   return render_template('results_chest.html',final_chest_pred=final_chest_pred)

if __name__ == '__main__':
   app.secret_key = ".."
   app.run(debug = True)
   #app.run()
