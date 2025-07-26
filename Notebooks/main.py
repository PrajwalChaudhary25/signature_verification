import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from grad_cam import display_gradcam, make_gradcam_heatmap
from utils import load_and_prep_image, get_last_conv_layer_name


model = tf.keras.models.load_model("best_model.keras")
print(model.summary())

forged_image_path = "test_dataset/forged/1.tif"
genuine_image_path = "test_dataset/genuine/5.tif"

img, img_array = load_and_prep_image(forged_image_path)
prediction = model.predict(img_array)[0][0]
class_labels = {0: 'Forgery', 1: 'Genuine'}

prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)
predicted_label = class_labels[predicted_class_index]

# Display the image with labels
plt.imshow(img,cmap='gray')
plt.title(f"Predicted: {predicted_label}")  # actual_label needs to be obtained
plt.axis('off')  
plt.show()


last_conv_layer_name = get_last_conv_layer_name(model=model)
display_gradcam(forged_image_path, model=model, last_conv_layer_name=last_conv_layer_name)