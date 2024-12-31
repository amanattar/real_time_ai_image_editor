import streamlit as st
from style_transfer import style_transfer
from noise_reduction import noise_reduction
from image_super_resolution import super_resolve

st.set_page_config(page_title="Real-Time AI Image Editor", layout="centered")

# Home Page
title = '<p style="text-align: center;font-size: 50px;font-weight: 350;font-family:Cursive "> AI Image Editor </p>'
st.markdown(title, unsafe_allow_html=True)

st.markdown(
    "<b><i>Transform Your Images with AI Magic!</i></b> &nbsp; Use cutting-edge AI techniques to apply artistic styles, reduce noise, and enhance image resolution effortlessly.",
    unsafe_allow_html=True
)

 # line break
st.markdown(" ")
# About the programmer
st.markdown("## Made by **Aman Attar** \U0001F609")
st.markdown("Contribute to this project at "
            "[*github.com/amanattar*](https://github.com/amanattar/real_time_ai_image_editor)")



st.write("Choose one of the options below:")

# Options
option = st.selectbox("Choose a feature:", ["Select", "Style Transfer", "Noise Reduction", "Image Super-Resolution"])

if option == "Style Transfer":
    st.header("Style Transfer")
    st.write("Upload Content and Style images to create a stylized image.")
    col1, col2 = st.columns(2)

    with col1:
        content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
    with col2:
        style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

    if content_file and style_file:
        with st.spinner("Stylizing..."):
            result_image = style_transfer(content_file, style_file)
        st.image(result_image, caption="Stylized Image", width=500)
        st.download_button("Download Stylized Image", result_image, "stylized_image.png")

elif option == "Noise Reduction":
    st.header("Noise Reduction")
    st.write("Upload an image to reduce noise.")
    file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if file:
        with st.spinner("Reducing noise..."):
            result_image = noise_reduction(file)
        st.image(result_image, caption="Denoised Image", width=500)
        st.download_button("Download Denoised Image", result_image, "denoised_image.png")

elif option == "Image Super-Resolution":
    st.header("Image Super-Resolution")
    st.write("Upload an image to upscale to high resolution.")
    file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if file:
        with st.spinner("Upscaling image..."):
            result_image = super_resolve(file)
        st.image(result_image, caption="Super-Resolved Image", width=500)
        st.download_button("Download Super-Resolved Image", result_image, "super_resolved_image.png")
