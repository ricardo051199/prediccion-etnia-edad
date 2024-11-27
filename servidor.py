from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # Permitir todas las conexiones CORS

@app.route('/upload', methods=['POST'])
def upload():
    datos = request.get_json()
    imagen_base64 = datos['imagen']
    imagen_bytes = base64.b64decode(imagen_base64)
    np_arr = np.frombuffer(imagen_bytes, np.uint8)
    imagen = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Procesar la imagen (escala de grises)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    print(imagen_gris.shape)
    cv2.imshow("Digitos", imagen_gris)
    cv2.waitKey()
    _, buffer = cv2.imencode('.jpg', imagen_gris)
    imagen_gris_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'imagen_procesada': imagen_gris_base64}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
