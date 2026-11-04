"""
Cargar modelo desde MLflow en producción
"""

import mlflow
import mlflow.keras
import pickle
import json
import numpy as np

class PredictorMLflow:
    """Carga modelos desde MLflow"""

    def __init__(self, model_name='predictor-precios', version='latest'):
        """
        Args:
            model_name: Nombre registrado en MLflow
            version: 'latest', '1', '2', etc
        """
        self.model_name = model_name
        self.version = version

        mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

        model_uri = f"models:/{model_name}/{version}"
        print(f"📦 Cargando: {model_uri}")

        self.modelo = mlflow.keras.load_model(model_uri)

        run_id = self._get_run_id()
        self.scaler = self._cargar_artefacto('scaler.pkl', run_id)
        self.info = self._cargar_artefacto_json('info_modelo.json', run_id)

        print(f"✅ Modelo cargado exitosamente")
        print(f"   Versión: {self.info.get('version')}")
        print(f"   MAE: ${self.info['metricas']['mae']:.2f}\n")

    def _get_run_id(self):
        """Obtiene run_id del modelo registrado"""
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{self.model_name}'")
        if self.version == 'latest':
            latest = max(versions, key=lambda x: int(x.version))
            return latest.run_id
        else:
            for v in versions:
                if v.version == str(self.version):
                    return v.run_id
        raise ValueError(f"Versión {self.version} no encontrada")

    def _cargar_artefacto(self, nombre, run_id):
        """Carga archivo pickle desde MLflow"""
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, nombre)
        with open(local_path, 'rb') as f:
            return pickle.load(f)

    def _cargar_artefacto_json(self, nombre, run_id):
        """Carga JSON desde MLflow"""
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, nombre)
        with open(local_path) as f:
            return json.load(f)

    def predecir(self, medinc, houseage, averrooms, avebedrooms,
                 population, aveoccup, latitude, longitude):
        """Predicción"""
        datos = np.array([[medinc, houseage, averrooms, avebedrooms,
                          population, aveoccup, latitude, longitude]])

        datos_norm = self.scaler.transform(datos)
        precio = self.modelo.predict(datos_norm, verbose=0)[0][0] * 100000

        return {
            'precio': round(precio, 2),
            'error_estimado': self.info['metricas']['mae'],
            'rango': {
                'minimo': round(precio - self.info['metricas']['mae'], 2),
                'maximo': round(precio + self.info['metricas']['mae'], 2)
            }
        }


if __name__ == '__main__':
    predictor = PredictorMLflow(model_name='predictor-precios', version='latest')

    resultado = predictor.predecir(
        medinc=8.0,
        houseage=20,
        averrooms=6,
        avebedrooms=1,
        population=1000,
        aveoccup=3,
        latitude=37.8,
        longitude=-122.2
    )

    print(f"🏠 Predicción:")
    print(f"   Precio: ${resultado['precio']:,.2f}")
    print(f"   Rango: ${resultado['rango']['minimo']:,.0f} - ${resultado['rango']['maximo']:,.0f}")
