from keras.engine.saving import model_from_json
from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import os

WORK_DIR = '/kaggle/input/cassava-leaf-disease-classification'
MODEL_ARCHITECTURE = '/kaggle/input/models/your_model.json'
MODEL_WEIGHTS = '/kaggle/input/models/your_model.h5'
TARGET_SIZE = 512

with open(MODEL_ARCHITECTURE, 'r') as f:
    model = model_from_json(f.read())
model.load_weights(MODEL_WEIGHTS)

# write CSV    
predictions = {"image_id": [], "label": []}
test_data = os.path.join(WORK_DIR, "test_images")
test_images = os.listdir(test_data)

for file_name in test_images:
    image = Image.open(os.path.join(WORK_DIR,  "test_images", file_name))
    image = image.resize((TARGET_SIZE, TARGET_SIZE))
    image = np.expand_dims(image, axis = 0)
    predictions['image_id'].append(file_name)
    pred = model.predict(image)
    predictions['label'].append(np.argmax(pred, axis=-1)[0])

pd.DataFrame(predictions).to_csv('submission.csv', index=False)
