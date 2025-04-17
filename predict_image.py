import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
import cv2

# Завантаження моделі
model = tf.keras.models.load_model("inception_pet_classifier.h5")

# Завантаження назв класів
if os.path.exists("class_names.json"):
    with open("class_names.json", "r") as f:
        classes = json.load(f)
else:
    classes = ['cat', 'dog']

# Завантаження каскаду Хаара для виявлення тіл (як варіант — собак)
# Можна спробувати haarcascade_frontalcatface.xml або haarcascade_frontalface_default.xml
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)

def predict_image(img_path):
    # Завантаження зображення для класифікації
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Передбачення
    predictions = model.predict(x)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    raw_label = classes[class_index]
    breed_name = raw_label.split('-')[-1] if '-' in raw_label else raw_label
    label = f"{breed_name} ({confidence * 100:.1f}%)"  # Виводимо як відсоток

    print(f"Результат: {label}")

    # Завантаження оригінального зображення для OpenCV
    img_cv = cv2.imread(img_path)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Виявлення об'єктів
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(boxes) > 0:
        for (x, y, w, h) in boxes:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_cv, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    else:
        # Якщо не знайдено — просто виводимо назву породи у верхній частині
        cv2.putText(img_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Result", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_image("examples/doberman #3.jpg")