import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Константи
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "dataset"

# Генератор з аугментацією
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Генератори для тренування та валідації
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Збереження назв класів
class_names = list(train_generator.class_indices.keys())
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Модель InceptionV3 без верхнього шару
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Заморожуємо ваги базової моделі
for layer in base_model.layers:
    layer.trainable = False

# Додаємо свої шари
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Компіляція
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Навчання
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Збереження моделі
model.save("inception_pet_classifier.h5")

# Візуалізація точності
plt.plot(history.history['accuracy'], label='Тренування')
plt.plot(history.history['val_accuracy'], label='Валідація')
plt.title('Точність')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.show()