import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)
model = load_model('my_model.h5')

def preprocess_image(image_path):
    # Load the image and preprocess it
    image = Image.open(image_path)
    # Convert to grayscale
    image = image.convert('L')
    # Resize the image
    image = image.resize((256, 256))
    # Convert to array
    image = img_to_array(image)
    # Expand dimensions to match model's input shape
    image = np.expand_dims(image, axis=0)
    # Normalize pixel values
    image /= 255.0
    return image


# Route for home page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

class_indices = {'glioma': 0, 'meningioma': 1, 'notumar': 2, 'pituitary': 3}

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Save the image to a temporary location
    imagefile = request.files['imagefile']
    image_path = 'images/' + imagefile.filename
    imagefile.save(image_path)
    
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Make predictions
    prediction = model.predict(image)
    
  
    class_index = np.argmax(prediction, axis=1)
    
     # Get predicted class index
    predicted_index = np.argmax(prediction, axis=1)[0]
    
    # Get predicted class label
    predicted_label = list(class_indices.keys())[list(class_indices.values()).index(predicted_index)]
    
    return render_template('result.html', class_label=predicted_label)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
