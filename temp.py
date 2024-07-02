from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
import numpy as np

# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')
model = load_model('leaf_disease_model.h5')

@app.route('/')
def home():
    return render_template('tempindex.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = image.load_img(BytesIO(file.read()), target_size=(224, 224))  # Load image from BytesIO
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        # Assuming you have a list of class names
        class_names = ['Apple___Apple_scab Fertilizer - Zinc sulfate and urea', 'Apple___Black_rot Fertilizer - Mancozeb and Ziram',
                       'Apple___Cedar_apple_rust fertilizer - Myclobutani','Apple__healthy','Blueberry_healthy',
                       'Cherry_(including_sour)__healthy','Cherry_(including_sour)___Powdery_mildew Fertilizer - Nimrod',
                       'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot Fertilizer - SYSTHANE (myclobutanil) and',
                       'Corn_(maize)___Common_rust_  Fertilizer - Azoxystrobin','Corn_(maize)__healthy',
                       'Corn_(maize)___Northern_Leaf_Blight Fertilizer - azoxystrobin, picoxystrobin','Grape___Black_rot Fertlizer - Fungicides',
                       'Grape___Esca_(Black_Measles) Fertilizer - Calcium Nitrate,10-10-10 (N-P-K),Epsom Salt (Magnesium Sulfate)','Grape___healthy',
                       'Grape___Leaf_blight_(Isariopsis_Leaf_Spot) Fertilizer - Micronutrient Mixes,Calcium Nitrate',
                       'Orange___Haunglongbing_(Citrus_greening) Fertilizer - neonicotinoids imidacloprid,thiamethoxam and clothianidin',
                       'Peach___Bacterial_spot Fertilizer - oxytetracycline','Peach___healthy','Pepper,_bell___Bacterial_spot Fertilizer - Copper sprays',
                       'Pepper,_bell___healthy','Potato___Early_blight Fertlizer - High nitrogen and low phosphorous','Potato___healthy',
                       'Potato___Late_blight Fertilizer - Dithane (mancozeb) MZ','Squash___Powdery_mildew Fertlizer - jojoba oil (e.g., E-rase), suphur','Strawberry___healthy',
                       'Strawberry___Leaf_scorch Fertilizer - chlorothalonil, myclobutanil or triflumizole','Tomato___Bacterial_spot Fertlizer - Copper-containing bactericides',
                       'Tomato___Early_blight Fertilizer - mandipropamid, chlorothalonil, fluazinam,mancozeb','Tomato___healthy',
                       'Tomato___Late_blight fertilizer - chlorothalonil (Bravo, Echo, Equus, or Daconil) and Mancozeb (Manzate)',
                       'Tomato___Leaf_Mold fertilizer - Copper-based fungicides','Tomato___Septoria_leaf_spot Fertilizer - chlorothalonil and mancozeb',
                       'Tomato___Spider_mites Two-spotted_spider_mite Fertilizer - Miticides,hexythiazox','Tomato___Target_Spot fertilizer - chlorothalonil, copper oxychloride or mancozeb',
                       'Tomato___Tomato_mosaic_virus fertilizer - Chlorothalonil','Tomato___Tomato_Yellow_Leaf_Curl_Virus Fertilizer - Imidacloprid']
        disease = class_names[predicted_class]
        return render_template('result.html', disease=disease)

if __name__ == '__main__':
    app.run(debug=True)