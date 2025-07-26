import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from grad_cam import display_gradcam, make_gradcam_heatmap
from utils import load_and_prep_image, get_last_conv_layer_name


model = tf.keras.models.load_model("best_model.keras")
print(model.summary())

forged_image_path = "test_dataset/forged/1.tif"
genuine_image_path = "test_dataset/genuine/25.tif"

img, img_array = load_and_prep_image(forged_image_path)
prediction = model.predict(img_array)[0][0]
print(prediction)
if prediction >= 0.5:
    print(f"Prediction: Genuine (confidence: {prediction:.2f})")
else:
    print(f"Prediction: Forged (confidence: {1 - prediction:.2f})")

last_conv_layer_name = get_last_conv_layer_name(model=model)
display_gradcam(forged_image_path, model=model, last_conv_layer_name=last_conv_layer_name)