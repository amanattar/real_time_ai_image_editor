import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import io

def transfer_style(content_image_file, style_image_file, model_path):
    """
    :param content_image_file: file-like object of the content image
    :param style_image_file: file-like object of the style image
    :param model_path: path to the downloaded pre-trained model.
    :return: An image as a 3D numpy array.
    """

    print("Loading images...")
    # Open content and style images as PIL objects
    content_image = Image.open(content_image_file).convert("RGB")
    style_image = Image.open(style_image_file).convert("RGB")

    # Convert images to numpy arrays
    content_image = np.array(content_image)
    style_image = np.array(style_image)

    print("Resizing and Normalizing images...")
    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    # Resize style image to 256x256 (recommended size for the model)
    style_image = tf.image.resize(style_image, (256, 256))

    print("Loading pre-trained model...")
    # Load the TensorFlow Hub model
    hub_module = hub.load(model_path)

    print("Generating stylized image now...wait a minute")
    # Stylize the content image with the style image
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    # Convert Tensor to a numpy array and reshape
    stylized_image = np.array(stylized_image)
    stylized_image = stylized_image.reshape(
        stylized_image.shape[1], stylized_image.shape[2], stylized_image.shape[3]
    )

    print("Stylizing completed...")
    return stylized_image
