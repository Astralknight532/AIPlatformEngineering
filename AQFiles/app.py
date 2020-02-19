from flask import Flask, render_template, request
import tensorflow as tf
import keras
from keras.models import load_model

app = Flask(__name__)

# Loading the models
no2_mod = load_model('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/no2_model.h5')
so2_mod = load_model('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/so2_model.h5')
o3_mod = load_model('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/o3_model.h5')
co_mod = load_model('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/co_model.h5')

@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict_result():
    date = request.form["dateentry"] # Get the date entered by the user
    pol = request.form["polselect"] # Get the pollutant chosen by the user 
    avgconc = 0 # Create a variable to store the predicted avg. concentration for a pollutant

    # Select the appropriate model based on the user's chosen pollutant
    if pol == 'NO2':
        print('NO2 Model')
    elif pol == 'SO2':
        print('SO2 Model')
    elif pol == 'O3':
        print('O3 Model')
    elif pol == 'CO':
        print('CO Model')

    avgconc_print = str(avgconc)
    if pol == 'NO2' or pol == 'SO2':
        avgconc_print += ' parts per billion'
    elif pol == 'O3' or pol == 'CO':
        avgconc_print += ' parts per million'
    
    return render_template("results.html", chosendate = date, pollutant = pol, avgconc = avgconc_print)

app.run(debug = True)