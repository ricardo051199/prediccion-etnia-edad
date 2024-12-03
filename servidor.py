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
    gender_model = tf.keras.models.load_model('gender_model.h5')
    age_model = tf.keras.models.load_model("modelo_prediccion_edad.h5", custom_objects={'mse': MeanSquaredError()})
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def save_image_array(image_array, file_path):
    # Eliminar las dimensiones extra y convertir a escala de grises
    image_2d = image_array[0, :, :, 0] * 255  # Quitar batch y normalización
    image_pil = Image.fromarray(image_2d.astype('uint8'))  # Convertir a formato PIL
    image_pil.save(file_path)


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

        # Guardar image_array como imagen
        processed_image_path = 'processed_image_array.jpg'
        save_image_array(image_array, processed_image_path)

        # Predicción
        predictions = gender_model.predict(image_array)
        predicted_class = np.argmax(predictions)  # Clase predicha

        # Mapear clase a etiqueta
        genders = {0: 'Hombre', 1: 'Mujer'}  # Define tus clases aquí
        predicted_gender = genders.get(predicted_class, 'unknown')

        # Predicción de edad
        age_predictions = age_model.predict(image_array)
        predicted_age = int(age_predictions[0]) 

        return jsonify({'gender': predicted_gender, 'age': predicted_age})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
