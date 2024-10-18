import streamlit as st
from PIL import Image
from flask_backend.models.image_classifier import classify_image
from flask_backend.models.text_classifier import classify_text

# Title and Explanation
st.title("🏡 Apartment Accessibility Checker")

# Add the image
image = Image.open("streamlit_application/Adaptable_House.jpg")
st.image(image, caption="Adaptable House for All", use_column_width=True)

# Brief explanation of the application
st.markdown("""
## 📝 Brief Explanation
This app evaluates the **accessibility** of an apartment for individuals with disabilities by analyzing both images and text descriptions.

Using **CLIP** from **Hugging Face's transformers**, it detects features like ramps, wide doorways, and step-free entrances in the images. It also uses a **zero-shot classification pipeline** to assess accessibility in the text.

You can upload images and provide a description, and the app will generate an accessibility score for each input and a **general confidence score** reflecting the overall accessibility.
""")

# Function to reset previous results
def reset_results():
    st.session_state.clear()  # Clear all session state to remove previous inputs and results

# Image Upload Section
st.header("📸 Upload Images")
uploaded_images = st.file_uploader("Choose one or more images to check for accessibility features", type=["jpg", "jpeg"], accept_multiple_files=True)

# Text Description Section
st.header("📝 Apartment Description")
description = st.text_area("Enter apartment description for accessibility analysis")

# Function to calculate the general confidence score
def calculate_general_confidence(image_scores, text_score=None):
    if image_scores and text_score is None:
        # If only images are provided
        return sum(image_scores) / len(image_scores)
    elif text_score is not None and not image_scores:
        # If only text is provided
        return text_score
    elif image_scores and text_score is not None:
        # If both images and text are provided, combine them (weighted average)
        return (0.6 * (sum(image_scores) / len(image_scores))) + (0.4 * text_score)
    else:
        return 0  # No images or text provided

# Button to trigger analysis
if st.button("🚀 Analyze"):
    st.markdown("### 🔎 Analysis Results")
    image_scores = []
    text_score = None

    # Process each uploaded image
    if uploaded_images:
        for uploaded_image in uploaded_images:
            # Display the image while it's being analyzed
            st.write(f"Analyzing Image: **{uploaded_image.name}**")

            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption=f"Analyzing {uploaded_image.name}...", use_column_width=True)

           # Call the classify_image function from models/image_classifier.py
            result = classify_image(uploaded_image)
            if result:
                for feature_info in result:
                    st.write(f"**Detected Feature:** {feature_info['feature']}, **Confidence Score:** {feature_info['score']:.2f}")
                    image_scores.append(feature_info['score'])
            else:
                st.write(f"No accessible features detected in {uploaded_image.name}.")

    # Process text description if provided
    if description:
        # Call the classify_text function from models/text_classifier.py
        text_result = classify_text(description)
        st.write(f"**Text Classification:** {text_result['labels'][0]}, **Confidence Score:** {text_result['scores'][0]:.2f}")
        text_score = text_result['scores'][0]

    # Calculate and display the general confidence score
    general_confidence = calculate_general_confidence(image_scores, text_score)
    st.write(f"### 🏆 General Confidence Score: **{general_confidence:.2f}**")

# Footer for contact and additional links
st.markdown("---")
st.markdown("#### Developed by [Ibtihel Nemri](https://github.com/ibtihelnemri?tab=repositories).")
st.markdown("💡 Visit the [GitHub repository](https://github.com/ibtihelnemri?tab=repositories) for more projects!")
