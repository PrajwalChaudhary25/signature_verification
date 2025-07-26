import tensorflow as tf
import numpy as np

def get_last_conv_layer_name(model):
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        
def load_and_prep_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array