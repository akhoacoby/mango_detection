# cnn_classifier.py
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === Settings ===
IMG_SIZE = 128  # resize images to 128x128
DATA_CSV = "mango_color_features.csv"
IMAGE_FOLDER = "mango_blocks"  # folder containing images

# === Load CSV with image paths and labels ===
df = pd.read_csv(DATA_CSV)
df.dropna(inplace=True)

# === Load Images and Labels ===
images = []
labels = []

for _, row in df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row["image"])
    img = cv2.imread(img_path)
    if img is None:
        continue  # skip if image not found
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    labels.append(row["color_label"])  # or use shape/quality if available

X = np.array(images) / 255.0
y = np.array(labels)

# === Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# === CNN Model ===
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(len(le.classes_), activation="softmax"),
    ]
)

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# === Train ===
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# === Evaluate ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# === Plot Accuracy ===
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Test the model with a sample image
def test_model(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0) / 255.0  # normalize and add batch dimension

    prediction = model.predict(img)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    print(f"Predicted label for {image_path}: {predicted_label[0]}")


if __name__ == "__main__":
    test_image_path = "mango\\dataset.v18i.tensorflow\\test\\1-2-_jpg.rf.1345445b10ba17bdbd26ea13448d1f7b.jpg"  # replace with your test image path
    test_model(test_image_path)
