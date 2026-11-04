"""
API Flask para predicción de precios
Ejecutar: python src/app.py
"""

from flask import Flask, request, jsonify
from predictor_mlflow import PredictorMLflow
import os

app = Flask(__name__)

# Cargar modelo al iniciar
try:
    predictor = PredictorMLflow(model_name='predictor-precios', version='latest')
except Exception as e:
    print(f"⚠️ No se pudo cargar el modelo: {e}")
    print("📌 Ejecuta primero: python src/entrenar.py")
    predictor = None

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'modelo': 'predictor-precios'})

@app.route('/info', methods=['GET'])
def info():
    """Información del modelo"""
    if predictor is None:
        return jsonify({'error': 'Modelo no cargado'}), 503
    return jsonify(predictor.info)

@app.route('/predecir', methods=['POST'])
def predecir():
    """
    Endpoint para predicción

    POST /predecir
    {
        "medinc": 8.0,
        "houseage": 20,
        "averrooms": 6,
        "avebedrooms": 1,
        "population": 1000,
        "aveoccup": 3,
        "latitude": 37.8,
        "longitude": -122.2
    }
    """
    try:
        if predictor is None:
            return jsonify({'error': 'Modelo no cargado'}), 503

        datos = request.json

        requeridas = ['medinc', 'houseage', 'averrooms', 'avebedrooms',
                     'population', 'aveoccup', 'latitude', 'longitude']

        if not all(k in datos for k in requeridas):
            return jsonify({'error': f'Campos requeridos: {requeridas}'}), 400

        resultado = predictor.predecir(
            medinc=datos['medinc'],
            houseage=datos['houseage'],
            averrooms=datos['averrooms'],
            avebedrooms=datos['avebedrooms'],
            population=datos['population'],
            aveoccup=datos['aveoccup'],
            latitude=datos['latitude'],
            longitude=datos['longitude']
        )

        return jsonify(resultado), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/lote', methods=['POST'])
def predecir_lote():
    """Predicción en lote (múltiples casas)"""
    try:
        if predictor is None:
            return jsonify({'error': 'Modelo no cargado'}), 503

        datos_lista = request.json

        resultados = []
        for datos in datos_lista:
            resultado = predictor.predecir(
                medinc=datos['medinc'],
                houseage=datos['houseage'],
                averrooms=datos['averrooms'],
                avebedrooms=datos['avebedrooms'],
                population=datos['population'],
                aveoccup=datos['aveoccup'],
                latitude=datos['latitude'],
                longitude=datos['longitude']
            )
            resultados.append(resultado)

        return jsonify({'predicciones': resultados}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    puerto = int(os.environ.get('PORT', 5001))
    print(f"🚀 Servidor corriendo en http://localhost:{puerto}")
    app.run(host='0.0.0.0', port=puerto, debug=False)
