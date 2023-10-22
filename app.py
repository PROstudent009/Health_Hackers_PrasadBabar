from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
from keras.models import load_model
import os
from flask import Flask, flash, request, redirect, url_for,jsonify



from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from flask import Flask, render_template, request

import tempfile
app=Flask(__name__)

result  = {0:"Acne and Rosacea Photos",1:"Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",2:
    "Atopic Dermatitis Photos",3:"Bullous Disease Photos",4:"Cellulitis Impetigo and other Bacterial Infections",5:"Eczema Photos",6:"Exanthems and Drug Eruptions",7:"Hair Loss Photos Alopecia and other Hair Diseases", 8:"Herpes HPV and other STDs Photos", 
           9: "Light Diseases and Disorders of Pigmentation", 10: "Lupus and other Connective Tissue diseases", 11: "Melanoma Skin Cancer Nevi and Moles", 12: "Nail Fungus and other Nail Disease", 13: "Poison Ivy Photos and other Contact Dermatitis", 14: "Psoriasis pictures Lichen Planus and related diseases", 15: "Scabies Lyme Disease and other Infestations and Bites", 
           16: "Seborrheic Keratoses and other Benign Tumors", 17: "Systemic Disease", 18: "Tinea Ringworm Candidiasis and other Fungal Infections", 19: "Urticaria Hives", 20: "Vascular Tumors", 21: "Vasculitis Photos", 22: "Warts Molluscum and other Viral Infections"}
MODEL_PATH =   'C:\SIH\My_Model.h5'

model_dl = load_model(MODEL_PATH)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img_file = request.files['image']
        
        # Save the temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            img_file.save(temp.name)
            temp_path = temp.name
        
        # Preprocess the image for the model
        img = image.load_img(temp_path, target_size=(224, 224))
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = np.vstack([img_arr])
        # Make a prediction with the model
        Result = model_dl.predict(img_arr)
        prediction_index = np.argmax(Result)
        prediction = result[prediction_index]
        
        # Remove the temporary file
        os.remove(temp_path)
        
        # Return the prediction as a JSON response
        response = {'prediction': str(prediction)}
        return jsonify(response)
    else:
        return "No image passed"


if __name__== '__main__':
    app.run( )
    
    