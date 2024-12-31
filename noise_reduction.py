import cv2
import numpy as np
from PIL import Image
import io

def noise_reduction(file):
    # Load the image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Apply noise reduction
    noiseless = cv2.fastNlMeansDenoisingColored(image, None, h=20, hColor=20, templateWindowSize=7, searchWindowSize=21)

    # Convert to RGB for display and download
    noiseless = cv2.cvtColor(noiseless, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(noiseless)

    # Save to BytesIO
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer
