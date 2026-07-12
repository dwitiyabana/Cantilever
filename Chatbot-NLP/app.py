from flask import Flask, render_template, request
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = load_model("chatbot_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("responses.pkl", "rb") as f:
    responses = pickle.load(f)

maxlen = model.input_shape[1]


def get_response(message):
    seq = tokenizer.texts_to_sequences([message.lower()])
    seq = pad_sequences(seq, maxlen=maxlen)

    pred = model.predict(seq, verbose=0)

    index = np.argmax(pred)

    return responses[index]


@app.route("/", methods=["GET", "POST"])
def home():

    bot_reply = ""

    if request.method == "POST":
        user = request.form["message"]
        bot_reply = get_response(user)

    return render_template("index.html", reply=bot_reply)


if __name__ == "__main__":
    app.run(debug=True)