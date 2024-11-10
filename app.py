import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model



# Load the pre-trained model
model = load_model('model.h5')

def preprocess_image(image):
    # Convert the uploaded image to grayscale and resize it to 28x28 pixels
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    
    # Convert image to numpy array and normalize pixel values to [0,1]
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input

    return img_array

def predict_class(model, img_array):
    # Get model prediction for the input
    prediction = model.predict(img_array)
    
    # Get the class index with the maximum probability
    predicted_index = np.argmax(prediction)
    
    return predicted_index

def main():
    st.title('MNIST Alphabet Classifier')

    # File uploader for grayscale images
    uploaded_file = st.file_uploader("Upload a grayscale image of any size", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict the class
        predicted_label_index = predict_class(model, processed_image)

        # Convert the index to the corresponding letter (assuming 'A' starts at index 0)
        predicted_alphabet = chr(predicted_label_index + ord("A"))

        # Display the prediction result
        st.success(f'The predicted alphabet is: {predicted_alphabet}')

if __name__ == "__main__":
    main()
