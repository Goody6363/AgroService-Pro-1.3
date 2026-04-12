import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

# Проверяем версию TensorFlow
print("TensorFlow version:", tf.__version__)  # должен быть 2.13.0

IMG_SIZE = 224
dataset_path = "database"

images = []
gear_labels = []
damage_labels = []

gear_map = {}
damage_map = {}

gear_idx = 0
damage_idx = 0

# ---------- ЗАГРУЗКА ДАННЫХ ----------
for gear in os.listdir(dataset_path):
    gear_path = os.path.join(dataset_path, gear)
    if not os.path.isdir(gear_path):
        continue

    gear_map[gear] = gear_idx
    gear_idx += 1

    for damage in os.listdir(gear_path):
        damage_path = os.path.join(gear_path, damage)
        if not os.path.isdir(damage_path):
            continue

        if damage not in damage_map:
            damage_map[damage] = damage_idx
            damage_idx += 1

        for img_file in os.listdir(damage_path):
            if not img_file.lower().endswith(("jpg", "jpeg", "png")):
                continue

            img_path = os.path.join(damage_path, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img) / 255.0

                images.append(img)
                gear_labels.append(gear_map[gear])
                damage_labels.append(damage_map[damage])

            except Exception as e:
                print("Ошибка при загрузке:", img_path, e)

images = np.array(images)
gear_labels = to_categorical(gear_labels, len(gear_map))
damage_labels = to_categorical(damage_labels, len(damage_map))

print("Всего изображений:", len(images))

# ---------- SPLIT ----------
X_train, X_test, y_gear_train, y_gear_test, y_damage_train, y_damage_test = train_test_split(
    images,
    gear_labels,
    damage_labels,
    test_size=0.2,
    random_state=42
)

# ---------- СОЗДАНИЕ МОДЕЛИ ----------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)

gear_output = Dense(len(gear_map), activation="softmax", name="gear")(x)
damage_output = Dense(len(damage_map), activation="softmax", name="damage")(x)

model = Model(inputs=base_model.input, outputs=[gear_output, damage_output])

model.compile(
    optimizer="adam",
    loss={
        "gear": "categorical_crossentropy",
        "damage": "categorical_crossentropy"
    },
    metrics=["accuracy"]
)

# ---------- ОБУЧЕНИЕ ----------
model.fit(
    X_train,
    {
        "gear": y_gear_train,
        "damage": y_damage_train
    },
    validation_data=(
        X_test,
        {
            "gear": y_gear_test,
            "damage": y_damage_test
        }
    ),
    epochs=10,
    batch_size=8
)

# ---------- СОХРАНЕНИЕ МОДЕЛИ В H5 ----------
model.save("gear_model.h5")  # <- единый .h5 файл для безопасного деплоя

# ---------- СОХРАНЕНИЕ MAP ----------
with open("gear_map.json", "w", encoding="utf-8") as f:
    json.dump(gear_map, f, ensure_ascii=False, indent=2)

with open("damage_map.json", "w", encoding="utf-8") as f:
    json.dump(damage_map, f, ensure_ascii=False, indent=2)

print("✅ МОДЕЛЬ ОБУЧЕНА И СОХРАНЕНА В gear_model.h5")