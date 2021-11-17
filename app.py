
from __future__ import division, print_function
# coding=utf-8
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

import sys
import os
import glob
import re
import numpy as np
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,redirect,url_for,request,render_template
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app =Flask(__name__)


model_path ='mobilenet_after_tune.h5'

model = load_model(model_path)

#model.make_predict_function() 

#preprocessing the step
def model_predict(img_path,model):
    
    img = image.load_img(img_path,target_size = (160,160,3))
    

    img = image.img_to_array(img)
    img = tf.expand_dims(img, 0)
    #img_pred = image.img_to_array(img)
    #img_pred = np.expand_dims(x,axis=0)
    #img = np.expand_dims(img_pred,axis =0)
     
    #img=preprocess_input(img)
    prediction = model.predict(img)

    # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #prediction = model.predict_classes(img)
    return prediction
  


	# Constants:


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def upload():
    classes = ['abhay_deol','adil_hussain','ajay_devgn',                            'akshay_kumar',
                            'akshaye_khanna',
                            'amitabh_bachchan',
                            'amjad_khan',
                            'amol_palekar',
                            'amole_gupte',
                            'amrish_puri',
                            'anil_kapoor',
                            'annu_kapoor',
                            'anupam_kher',
                            'anushka_shetty',
                            'arshad_warsi',
                            'aruna_irani',
                            'ashish_vidyarthi',
                            'asrani',
                            'atul_kulkarni',
                            'ayushmann_khurrana',
                            'boman_irani',
                            'chiranjeevi',
                            'chunky_panday',
                            'danny_denzongpa',
                            'darsheel_safary',
                            'deepika_padukone',
                            'deepti_naval',
                            'dev_anand',
                            'dharmendra',
                            'dilip_kumar',
                            'dimple_kapadia',
                            'farhan_akhtar',
                            'farida_jalal',
                            'farooq_shaikh',
                            'girish_karnad',
                            'govinda',
                            'gulshan_grover',
                            'hrithik_roshan',
                            'huma_qureshi',
                            'irrfan_khan',
                            'jaspal_bhatti',
                            'jeetendra',
                            'jimmy_sheirgill',
                            'johnny_lever',
                            'kader_khan',
                            'kajol',
                            'kalki_koechlin',
                            'kamal_haasan',
                            'kangana_ranaut',
                            'kay_kay_menon',
                            'konkona_sen_sharma',
                            'kulbhushan_kharbanda',
                            'lara_dutta',
                            'madhavan',
                            'madhuri_dixit',
                            'mammootty',
                            'manoj_bajpayee',
                            'manoj_pahwa',
                            'mehmood',
                            'mita_vashisht',
                            'mithun_chakraborty',
                            'mohanlal',
                            'mohnish_bahl',
                            'mukesh_khanna',
                            'mukul_dev',
                            'nagarjuna_akkineni',
                            'nana_patekar',
                            'nandita_das',
                            'nargis',
                            'naseeruddin_shah',
                            'navin_nischol',
                            'nawazuddin_siddiqui',
                            'neeraj_kabi',
                            'nirupa_roy',
                            'om_puri',
                            'pankaj_kapur',
                            'pankaj_tripathi',
                            'paresh_rawal',
                            'pawan_malhotra',
                            'pooja_bhatt',
                            'prabhas',
                            'prabhu_deva',
                            'prakash_raj',
                            'pran',
                            'prem_chopra',
                            'priyanka_chopra',
                            'raaj_kumar',
                            'radhika_apte',
                            'rahul_bose',
                            'raj_babbar',
                            'raj_kapoor',
                            'rajat_kapoor',
                            'rajesh_khanna',
                            'rajinikanth',
                            'rajit_kapoor',
                            'rajkummar_rao',
                            'rajpal_yadav',
                            'rakhee_gulzar',
                            'ramya_krishnan',
                            'ranbir_kapoor',
                            'randeep_hooda',
                            'rani_mukerji',
                            'ranveer_singh',
                            'ranvir_shorey',
                            'ratna_pathak_shah',
                            'rekha',
                            'richa_chadha',
                            'rishi_kapoor',
                            'riteish_deshmukh',
                            'sachin_khedekar',
                            'saeed_jaffrey',
                            'saif_ali_khan',
                            'salman_khan',
                            'sanjay_dutt',
                            'sanjay_mishra',
                            'shabana_azmi',
                            'shah_rukh_khan',
                            'sharman_joshi',
                            'sharmila_tagore',
                            'shashi_kapoor',
                            'shreyas_talpade',
                            'smita_patil',
                            'soumitra_chatterjee',
                            'sridevi',
                            'sunil_shetty',
                            'sunny_deol',
                            'tabu',
                            'tinnu_anand',
                            'utpal_dutt',
                            'varun_dhawan',
                            'vidya_balan',
                            'vinod_khanna',
                            'waheeda_rehman',
                            'zarina_wahab',
                            'zeenat_aman']

   
    


    if request.method=="POST":
        ##get the file from the post
        f = request.files['file']
        #save the file uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        prediction = model_predict(file_path, model)
        predicted_class = classes[np.argmax(prediction[0])]
        confidence = tf.nn.softmax(prediction[0])
        confidence = 100 * np.max(confidence)
        confidence = round(confidence,2)
        predicted_class= str(predicted_class.lower())
        print('We think that is {}.'.format(predicted_class.lower()))
        #actual_class = class_names[labels[i]] 
        return (f"Predicted celebrity  seems to be : {predicted_class},  With Confidence: {confidence}%")
        #return ('Actor sems to be {}.','with confidence'.format(predicted_class.lower()),confidence)
    #return None

        
if __name__=='__main__':
    app.run(debug=True)
