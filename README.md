# Real-Time AI Image Editor

## Overview

<b><i>Transform Your Images with AI Magic!</i></b> &nbsp; This project allows you to apply cutting-edge AI techniques to your images. The tool supports:

1. **Style Transfer**: Combine a content image and a style image to generate a unique stylized output.
2. **Noise Reduction**: Remove unwanted noise from your images while retaining details.
3. **Image Super-Resolution**: Enhance low-resolution images to high resolution with impressive clarity.

Creator: **Aman Attar**

LinkedIn: [Aman Attar](https://www.linkedin.com/in/amanattar/)

---

## Features

### 1. Style Transfer
- **What it does**: Allows you to blend two images by applying the artistic style of one image to the content of another. The result retains the main elements of the content image but looks "painted" in the style of the style image.
- **Use Case**: Create stunning artistic visuals by combining your photos with famous artwork styles.

### 2. Noise Reduction
- **What it does**: Removes noise or unwanted grainy textures from images while preserving the essential details.
- **Use Case**: Enhance clarity in photos taken in low-light conditions or reduce artifacts in scanned documents.

### 3. Image Super-Resolution
- **What it does**: Upscales low-resolution images to a higher resolution, adding detail and clarity to the image.
- **Use Case**: Improve the quality of old or small images for printing, presentations, or social media sharing.

---



## How to Run

### Prerequisites
Ensure you have conda install.

### 1. Clone the Repository
```bash
# Clone the repository
$ git clone https://github.com/amanattar/real_time_ai_image_editor
$ cd real_time_ai_image_editor
```

### 2. Creating Environment and Installing Dependencies
```bash
# Install required Python libraries
$ conda create --name ai_image_editor python=3.11
$ conda activate ai_image_editor
$ pip install -r requirements.txt
```

### 3. Run the Application
```bash
# Run the Streamlit app
$ streamlit run app.py
```

---

## How to Use

1. Open the Streamlit app in your browser (default: http://localhost:8501).
2. Choose one of the three options from the dropdown menu:
    - **Style Transfer**
    - **Noise Reduction**
    - **Image Super-Resolution**
3. Upload the required image(s) based on your selected option.
4. Click the process button to apply the AI technique.
5. View and download the output image.

---

## Example Use Cases

- **Style Transfer**: Create artistic visuals by blending famous artworks with your personal photos.
- **Noise Reduction**: Enhance the quality of scanned documents or low-light photos.
- **Image Super-Resolution**: Upgrade old or low-resolution photos to high resolution for printing or sharing.

---

## Models

The project uses pre-trained models for:
1. **Style Transfer**: TensorFlow Hub's arbitrary image stylization model.
2. **Noise Reduction**: OpenCV's Non-Local Means Denoising.
3. **Image Super-Resolution**: A PyTorch-based Residual-in-Residual Dense Block (RRDB) model.

Ensure that the models are downloaded and placed in the `models/` directory.

---

## Creator

- **Aman Attar**
- LinkedIn: [Aman Attar](https://www.linkedin.com/in/amanattar/)

Feel free to reach out for feedback, suggestions, or collaboration opportunities!

---

## Future Enhancements

- Add additional AI-powered features like image segmentation or object removal.
- Allow batch processing of images.
- Improve UI design for better user experience.

---

## License

This project is open-source and available under the Apache License 2.0. Feel free to use, modify, and distribute as per the terms of the license.

