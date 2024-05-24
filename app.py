import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load your pre-trained model
model = load_model('cats_vs_dogs_model.h5')

# Function to make a prediction
def predict(image_path, model):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction < 0.5:
        return 'Cat'
    else:
        return 'Dog'

# Streamlit app code
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Anurag-ghosh-12/Cats_Vs_Dogs_Classify/tree/main",
        "Report a bug": "https://github.com/Anurag-ghosh-12/Cats_Vs_Dogs_Classify/issues",
    },
)

# Streamlit app
st.title('Cat vs Dog Classifier')

# Set background color for left and right sections

# Sidebar with description
st.sidebar.title('About the Project')
st.sidebar.subheader(":blue[[Please use a desktop for the best experience.]]")
st.sidebar.info("""
This application was created by Anurag Ghosh using a Convolutional Neural Network (CNN) model trained on images of cats and dogs. The model can classify whether an uploaded image is of a cat or a dog with an accuracy of 84.22%.
""")


st.sidebar.image('https://images.pexels.com/photos/850602/pexels-photo-850602.jpeg?auto=compress&cs=tinysrgb&w=600', caption='Dog', use_column_width=True)
st.sidebar.image('https://images.pexels.com/photos/257532/pexels-photo-257532.jpeg?auto=compress&cs=tinysrgb&w=600', caption='Cat', use_column_width=True)
st.sidebar.link_button("[GitHub]", "https://github.com/Anurag-ghosh-12")
#Main content
st.link_button("[Dataset]", "https://www.kaggle.com/datasets/salader/dogs-vs-cats?select=train")
st.write('Upload an image of a cat or dog and the model will predict the class.')

# File uploader for image

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Save the uploaded image temporarily
    temp_image_path = 'temp_image.png'
    image.save(temp_image_path)

    # Make a prediction
    st.write('Classifying...')
    label = predict(temp_image_path, model)
    st.balloons()
    st.markdown(f'# This is a **{label}**!')
    st.image(image, caption='Uploaded Image', use_column_width=True)

#dummy images
expander = st.expander("Some real life images to try with...")
expander.write("Just drag-and-drop your chosen image above ")
example_images = [
    "./Dummies/dummy1.jpeg",
    "./Dummies/dummy2.jpeg",
    "./Dummies/dummy3.jpeg",
    "./Dummies/dummy4.jpeg",
    "./Dummies/dummy5.jpeg",
    "./Dummies/dummy6.jpeg",
]

num_columns = 3
rows = len(example_images) // num_columns + (1 if len(example_images) % num_columns else 0)
for row in range(rows):
    cols = expander.columns(num_columns)
    for col in range(num_columns):
        index = row * num_columns + col
        if index < len(example_images):
            cols[col].image(example_images[index], width=100)
#Accuracy graph
expander = st.expander("View Model Training and Validation Results")
expander.write("Confusion Matrix: ")
expander.image("./images/confusion_mat.png", use_column_width=True)
expander.write("Graphs: ")
expander.image("./images/plot.png", use_column_width=True)
# Footer
st.write("\n\n\n\n")
st.markdown("---")
st.markdown(
    f"""Drop in any discrepancies or give suggestions in `Report a bug` option within the `⋮` menu"""
)

st.markdown(
    f"""<div style="text-align: right"> Developed by Anurag Ghosh </div>""",
    unsafe_allow_html=True,
)
