from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import base64
import cv2
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.losses import MeanSquaredError

# Deshabilita GPU si no es necesaria
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
CORS(app)

# Carga del modelo
try:
    gender_model = tf.keras.models.load_model('modelos/gender_model.h5')
    age_model = tf.keras.models.load_model("modelos/age_model.h5", custom_objects={'mse': MeanSquaredError()})
    ethnicity_model = tf.keras.models.load_model("modelos/ethnicity_model.h5")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Procesar los datos recibidos
        data = request.get_json()
        image_data = data['image']
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')  # Convertir a RGB para OpenCV
        
        # Convertir la imagen a un formato compatible con OpenCV
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(open_cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({'error': 'No se detectaron rostros en la imagen.'}), 400

        # Tomar el primer rostro detectado
        x, y, w, h = faces[0]
        face = open_cv_image[y:y+h, x:x+w]

        # Convertir el rostro a escala de grises
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Preprocesar el rostro
        face_resized = cv2.resize(face_gray, (48, 48))  # Redimensionar a 48x48
        image_array = np.array(face_resized) / 255.0  # Normalizar los valores de los píxeles
        image_array = np.expand_dims(image_array, axis=-1)  # Añadir canal (48, 48, 1)
        image_array = np.expand_dims(image_array, axis=0)  # Añadir batch (1, 48, 48, 1)

        # Predicción
        predictions = gender_model.predict(image_array)
        predicted_class = np.argmax(predictions)  # Clase predicha

        # Mapear clase a etiqueta
        genders = {0: 'Hombre', 1: 'Mujer'}  # Define tus clases aquí
        predicted_gender = genders.get(predicted_class, 'unknown')

        # Predicción de edad
        age_predictions = age_model.predict(image_array)
        predicted_age = int(age_predictions[0]) 

        # Predicción de etnia
        ethnicity_predictions = ethnicity_model.predict(image_array)
        ethnicity_predicted_class = np.argmax(ethnicity_predictions)  # Clase predicha

        # Mapear clase a etiqueta
        ethnicity = {0: 'Caucásico', 1: 'Asiático', 2: 'Africano', 3: 'Latino'}  # Define tus clases aquí
        predicted_ethnicity = ethnicity.get(ethnicity_predicted_class, 'unknown')

        return jsonify({'gender': predicted_gender, 'age': predicted_age, 'ethnicity': predicted_ethnicity})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
