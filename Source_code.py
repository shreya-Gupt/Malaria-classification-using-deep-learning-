import json

# Load the notebook
with open("/mnt/data/Malaria.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Extract all code cells
code_cells = []
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        code_cells.append("".join(cell.get("source", [])))

code_output = "\n\n# -----------------------------\n\n".join(code_cells)

code_output[:5000]
DATA_DIR = "Malaria_Dataset"

# -----------------------------

import os
Parasitized_DIR = os.path.sep.join([DATA_DIR , "Parasitized"])
Unifected_DIR = os.path.sep.join([DATA_DIR , "Uninfected"])

# -----------------------------

len(os.listdir(Parasitized_DIR)), len(os.listdir(Unifected_DIR))

# -----------------------------

os.listdir(Parasitized_DIR)[:5]

# -----------------------------

os.listdir(Unifected_DIR)[:5]

# -----------------------------

import matplotlib.pyplot as plt 

# -----------------------------

import tensorflow as tf

# -----------------------------

from tensorflow.keras.utils import load_img , img_to_array

# -----------------------------

img = load_img(os.path.join(Parasitized_DIR , os.listdir(Parasitized_DIR)[0]) , target_size=(224 , 224))

plt.imshow(img)

# -----------------------------

plt.axis("off")

# -----------------------------

img

# -----------------------------

type(img)

# -----------------------------

plt.figure(figsize=(12 , 8))

parasitized_imgs = os.listdir(Parasitized_DIR)[:3]
uninfected_imgs = os.listdir(Unifected_DIR)[:3]

for i , img_name in enumerate(parasitized_imgs):
    img = load_img(os.path.join(Parasitized_DIR , img_name ) , target_size=(128 , 128))
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title("Parasitized")
    plt.axis("off")
    
for i , img_name in enumerate(uninfected_imgs):
    img = load_img(os.path.join(Unifected_DIR , img_name) , target_size=(128 , 128))
    plt.subplot(2, 3, i+4)
    plt.imshow(img)
    plt.title("Uninfected")
    plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------

from sklearn.model_selection import train_test_split
import numpy as np

# -----------------------------

data = []
labels = []

# - Parasitized

for name in os.listdir(Parasitized_DIR):
    img_path = os.path.join(Parasitized_DIR , name)
    img = load_img(img_path , target_size=(64 , 64))
    img_arr = img_to_array(img)
    data.append(img_arr)
    labels.append(1)
    
    
# - Uninfected
for name in os.listdir(Unifected_DIR):
    img_path = os.path.join(Unifected_DIR , name)
    img = load_img(img_path , target_size=(64 , 64))
    img_arr = img_to_array(img)
    data.append(img_arr)
    labels.append(0)

# -----------------------------

data = np.array(data , dtype="float32") / 255.0
labels = np.array(labels)

# -----------------------------

# Preparing Train and Test Data
X_train , X_test , y_train , y_test = train_test_split(data , labels , test_size=0.2 , random_state=42)

# -----------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Dense , Dropout

# -----------------------------

# Building CNN Model
model = Sequential([
    Conv2D(32 , (3,3) , activation="relu" , input_shape=(64 , 64 , 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64 , (3,3) , activation="relu"),
    MaxPooling2D((2,2)),
    
    Flatten(),
    
    Dense(128 , activation="relu"),
    Dropout(0.3),
    
    Dense(1 , activation="sigmoid")
])

# -----------------------------

model.compile(optimizer="adam" , loss="binary_crossentropy" , metrics=["accuracy"])

# -----------------------------

history = model.fit(X_train , y_train , validation_split=0.2 , epochs=7 , batch_size=32)

# -----------------------------

loss , accuracy = model.evaluate(X_test , y_test)

# -----------------------------

print("Test Loss:" , loss)
print("Test Accuracy:" , accuracy)

# -----------------------------

# Plot Accuracy

plt.figure(figsize=(10 , 4))
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# -----------------------------

# Plot Loss

plt.figure(figsize=(10 , 4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# -----------------------------

from sklearn.metrics import classification_report , confusion_matrix

preds = (model.predict(X_test) > 0.5).astype("int32")

print(classification_report(y_test , preds))

# -----------------------------

cm = confusion_matrix(y_test , preds)
cm

# -----------------------------

import seaborn as sns
plt.figure(figsize=(5 , 4))
sns.heatmap(cm , annot=True , fmt="d" , cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------

model.save("malaria_cnn_model.h5")

# -----------------------------

print("Model Saved Successfully")
