import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# Define a flask app
app = Flask(__name__)

# Load the saved model
model = load_model('cats_vs_dogs_model.h5', compile=False)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return prediction

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = model_predict(file_path, model)

        # Determine the predicted class
        result = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

        # Plotting the image with prediction
        img = Image.open(file_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f'Predicted: {result}')
        ax.axis('off')
        
        # Save the plot to a file
        plot_path = os.path.join(basepath, 'static', 'prediction.png')
        plt.savefig(plot_path)
        plt.close(fig)

        return render_template('predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
