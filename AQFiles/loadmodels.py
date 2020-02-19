from keras.models import load_model
from keras.models import model_from_json
from keras.optimizers import SGD
import tensorflow as tf
import keras.models

def loadno2():
    # Load NO2 model
    no2json = open('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/no2_model.json', 'r')
    no2loadedjson = no2json.read()
    no2json.close()
    no2loadedmod = model_from_json(no2loadedjson)

    # Load NO2 model's weights
    no2loadedmod.load_weights('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/no2_weights.h5')

    # Compile NO2 model
    opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
    no2loadedmod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
    no2graph = tf.compat.v1.get_default_graph()

    return no2loadedmod, no2graph

def loadso2():
    # Load SO2 model
    so2json = open('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/so2_model.json', 'r')
    so2loadedjson = so2json.read()
    so2json.close()
    so2loadedmod = model_from_json(so2loadedjson)

    # Load SO2 model's weights
    so2loadedmod.load_weights('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/so2_weights.h5')

    # Compile SO2 model
    opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
    so2loadedmod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
    so2graph = tf.compat.v1.get_default_graph()

    return so2loadedmod, so2graph

def loado3():
    # Load O3 model
    o3json = open('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/o3_model.json', 'r')
    o3loadedjson = o3json.read()
    o3json.close()
    o3loadedmod = model_from_json(o3loadedjson)

    # Load O3 model's weights
    o3loadedmod.load_weights('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/o3_weights.h5')

    # Compile O3 model
    opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
    o3loadedmod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
    o3graph = tf.compat.v1.get_default_graph()

    return o3loadedmod, o3graph

def loadco():
    # Load CO model
    co_json = open('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/co_model.json', 'r')
    co_loadedjson = co_json.read()
    co_json.close()
    co_loadedmod = model_from_json(co_loadedjson)

    # Load CO model's weights
    co_loadedmod.load_weights('C:/Users/hanan/Desktop/PersonalRepository/AQFiles/SavedModels/co_weights.h5')

    # Compile CO model
    opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
    co_loadedmod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
    co_graph = tf.compat.v1.get_default_graph()

    return co_loadedmod, co_graph