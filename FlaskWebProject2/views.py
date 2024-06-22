
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import os
from keras.models import load_model
from FlaskWebProject2 import app

# Загрузка обученной модели
model = load_model("FlaskWebProject2\model.h5")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Получение изображения
        image = request.files["image"]
        image_path = "FlaskWebProject2/static/uploads/" + image.filename
        image_path2 = "../static/uploads/" + image.filename
        image.save(image_path)

        # Преобразование изображения
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_data = (255.0 - img.reshape(-1)) / 255

        x = np.expand_dims(img_data, axis=0)
        prediction = model.predict(x)

        predicted_digit = np.argmax(prediction)

        return render_template(
            "result.html",
            digit=predicted_digit,
            image_path2=image_path2,
            image_path=image_path,
        )
    return render_template("index.html")


@app.route("/feedback", methods=["POST"])
def feedback():
    correct = request.form["correct"]
    predicted_digit = request.form["digit"]
    image_path = request.form["image_path"]
    if correct == "True":
        # Сохранение правильного распознанного изображения
        new_image_path = f"FlaskWebProject2/static/correct/recognized_{predicted_digit}_original_{image_path.split('/')[-1]}"
    else:
        # Сохранение неправильного распознанного изображения
        new_image_path = f"FlaskWebProject2/static/incorrect/recognized_{predicted_digit}_original_{image_path.split('/')[-1]}"

    if os.path.exists(new_image_path):
        os.remove(new_image_path)

    os.rename(image_path, new_image_path)

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
