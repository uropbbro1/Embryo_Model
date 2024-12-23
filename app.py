from types import prepare_class
import streamlit as st
import cv2
from joblib import load
import numpy as np
import tempfile
import keras

def load_model():
    model = keras.models.load_model('model.h5')
    return model

model = load_model()

# 2. Image Preprocessing (Adjust according to your model)
def preprocess_image(image):
    img = cv2.imread(image.name)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image,(110, 110), interpolation = cv2.INTER_AREA)
    gray_image = gray_image.reshape((110, 110, 1))
    return gray_image/255


# 3. Prediction Logic (Replace with your model's prediction)
def predict_embryo_quality(image):
    preprocessed_image = preprocess_image(image)

    # Assign class names based on the index.
    p_class = model.predict(np.array([preprocessed_image]))
    print(p_class)
    p_class_items = p_class.tolist()
    print(p_class_items)
    class_names = ["Bad", "Good"]
    if(p_class_items[0][0] > p_class_items[0][1]):
        confidence = p_class_items[0][0]
        index = 0
    else:
        confidence = p_class_items[0][1]
        index = 1
    print(index)
    print(class_names[index])
    predicted_class = class_names[index]
    #confidence_score = probabilities[0][predicted_class_index].item()

    return predicted_class, confidence

def main():
    st.title("Embryo Quality Prediction App")

    uploaded_file = st.file_uploader("Upload an embryo image", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Embryo Quality"):
           with st.spinner("Predicting quality..."):
             tfile = tempfile.NamedTemporaryFile(delete=False)
             tfile.write(uploaded_file.read())
             predicted_class, confidence_score = predict_embryo_quality(tfile)
             confidence_score = confidence_score * 100
             st.success("Prediction Complete!")
             st.write(f"Predicted quality: **{predicted_class}**")
             st.write(f"Confidence: {confidence_score:.4f}%")


if __name__ == "__main__":
    main()
