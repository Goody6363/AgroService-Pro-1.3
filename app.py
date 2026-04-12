from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
from flask_cors import CORS

import os

app = Flask(__name__)
CORS(app)

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model("gear_model.h5")

with open("drawings_map.json", "r", encoding="utf-8") as f:
    drawings_map = json.load(f)

gear_map = json.load(open("gear_map.json","r",encoding="utf-8"))
damage_map = json.load(open("damage_map.json","r",encoding="utf-8"))

gear_map = {int(v):k for k,v in gear_map.items()}
damage_map = {int(v):k for k,v in damage_map.items()}

repair_db = json.load(open("repair_db.json","r",encoding="utf-8"))

IMG_SIZE = 224

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)/255.0
    return np.expand_dims(img,0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]
    img = preprocess(file.read())

    gear_pred, damage_pred = model.predict(img)

    gear_id = int(np.argmax(gear_pred))
    damage_id = int(np.argmax(damage_pred))

    gear = gear_map[gear_id]
    damage = damage_map[damage_id]

    drawing = drawings_map.get(gear, "gear1.png")
    drawing = f"/static/drawings/{drawing}"

    key = f"{gear}_{damage}"

    result = repair_db.get(key, {
        "info": "Нет данных",
        "repair": "Инструкция отсутствует"
    })

    return jsonify({
        "gear": gear,
        "damage": damage,
        "info": result.get("info", "Нет данных"),
        "repair": result.get("repair", "Инструкция отсутствует"),
        "drawing": drawing
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
