import json

import numpy as np
import tensorflow as tf
import keras
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin

import tensorflow_addons as tfa
from tensorflow.python.ops.numpy_ops import np_config


np_config.enable_numpy_behavior()
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chain.sqlite"
cors = CORS(app)


def judge_under_model(image, model_type: str):
    model = keras.models.load_model('./checkpoints/{0}'.format(model_type))

    image_size = 128
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    # file_name = './realworld_test/must_be_ai_for_ADM.PNG'
    # image = tf.io.read_file(file_name)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.reshape(data_augmentation(image), (1, 128, 128, 3))
    print(image.shape)

    # Predict the class probabilities
    probabilities = list(model.predict(image))
    print(probabilities)

    # Get the predicted class index
    predicted_class_index = np.argmax(probabilities, axis=-1)
    print(predicted_class_index)

    return probabilities


# update implementation - Zackiss on 4.7bn

# ---------------FLASK part--------------------
@app.route('/')
@cross_origin()
def index():
    return render_template("index.html")


@app.route('/judge_image', methods=['POST', 'GET'])
@cross_origin()
def judge_image():
    """
    format: 127.0.0.1:5000/judge_image?pic=str
    """
    response = {}
    if len(request.args):
        response["status"] = 200
        response["info"] = "pic received"
        response["result"] = []
        pic = request.args.to_dict().get("pic", None)
        if pic is None:
            response["status"] = 300
            response["info"] = "transaction received with incorrect format"
        # if everything works well, we shall verify transaction
        else:
            response["result"] = judge_under_model(pic, model_type="ADM")
    else:
        response["status"] = 500
        response["info"] = "requesting with empty transaction"
    return json.dumps(response, ensure_ascii=False)


if __name__ == '__main__':
    app.run(port=9000)
