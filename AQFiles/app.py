from flask import Flask, render_template
import tensorflow as tf
import keras
from keras.models import load_model

app = Flask(__name__)

# Loading the models
no2_mod = load_model('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/savedmodels/no2_model.h5')
so2_mod = load_model('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/savedmodels/so2_model.h5')
o3_mod = load_model('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/savedmodels/o3_model.h5')
co_mod = load_model('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/savedmodels/co_model.h5')

@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict_result():
    return render_template("results.html")

app.run(debug = True)