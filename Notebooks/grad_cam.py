import tensorflow as tf
from utils import load_and_prep_image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # this context computes gradient of top predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, 0] 

    # Compute gradients of the class output with respect to conv layer output
    grads = tape.gradient(class_output, conv_outputs)

    # Mean intensity of gradients for each filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv layer output by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # this will normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam(img_path, model, last_conv_layer_name):
    original_img, prep_img = load_and_prep_image(img_path)
    
    heatmap = make_gradcam_heatmap(prep_img, model, last_conv_layer_name)

    # Convert grayscale to RGB to overlay heatmap
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize and colorize heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM Visualization")
    plt.show()
