from flask.globals import request
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.models import load_model
import os
from flask import Flask,render_template

model = load_model('Face_Mask.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST', 'GET'])
def predict():
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    img = load_img(os.path.join(app.config['UPLOAD_FOLDER'], f.filename), target_size = (224,224))
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)/255.0
    pred = model.predict(img)
    pred = pred[0,0]
    if pred < 0.5:
        prediction = 'Congrats!!! Mask is there'
    else:
        prediction = 'No Mask Detected'

    return render_template('index.html', prediction_text=f'Answer : {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
