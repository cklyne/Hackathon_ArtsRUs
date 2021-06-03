import os
import cv2
import tensorflow as tf
import numpy as np

import json
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin

SEED = 42
INP_SIZE = (224, 224)
MODEL_LOC = 'img_classifier_model_sigmoid'
LABELS_LOC = "label_trans.json"

app = Flask("model predictions")
CORS(app, support_credentials=True)


@app.route('/test_res', methods=['POST'])
def predict_single():
    """
    Predict based on json file passed through a post request
    :return: A list of the predicted values in a json file
    """
    imagefile = request.files.get('picture', '').read()
    npimg = np.fromstring(imagefile, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INP_SIZE)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    res = dict()
    for i, pred in enumerate(preds.reshape(-1)):
        res[label_trans[i]] = str(round(pred, 4))

    res = make_response(jsonify(res), 200)
    return res


def load_model(loc):
    """
    Loads a trained model
    :param loc: Location of the model
    :return: The loaded classification tf.keras model
    """
    if os.path.isdir(loc):
        new_model = tf.keras.models.load_model(loc, compile=True)
    else:
        raise ValueError("Did not find model in specified location")
    return new_model


def load_labels(loc):
    """
    Loads label dictionary for the model
    :param loc: Location of the labels
    :return: A dict translating between model output and index, and human readable label
    """
    with open(loc, 'r') as file:
        data = file.read()
    trn = json.loads(data)
    trn = {int(k): v for k, v in trn.items()}
    return trn


def main():
    app.run(host="0.0.0.0", port=80)


if __name__ == "__main__":
    model = load_model(MODEL_LOC)
    label_trans = load_labels(LABELS_LOC)
    main()
