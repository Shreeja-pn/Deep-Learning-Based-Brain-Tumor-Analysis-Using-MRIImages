import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Build classifier model
base_model = ResNet50(weights=None, include_top=False, input_shape=(128,128,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Load trained weights
model.load_weights("model/tumor_classifier_model.h5")

classes = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "Pituitary Tumor"
]


import nibabel as nib
import numpy as np
import cv2

def classify_tumor(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return classes[class_index]