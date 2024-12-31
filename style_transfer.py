import matplotlib.pyplot as plt
from PIL import Image
import io
from API import transfer_style  # Import updated transfer_style function

def style_transfer(content_file, style_file):
    # Define the path to the pre-trained model
    model_path = "models/model"

    # Call the transfer_style function with file-like objects
    result = transfer_style(content_file, style_file, model_path)

    # Save the result to a BytesIO buffer for Streamlit
    buffer = io.BytesIO()
    plt.imsave(buffer, result, format="PNG")
    buffer.seek(0)
    return buffer
