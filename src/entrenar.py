"""
Script de entrenamiento con MLflow
Ejecutar: python src/entrenar.py
"""

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# ============================================
# CONFIGURACIÓN DE MLflow
# ============================================

# Usar SQLite en lugar de file store (requerido en MLflow 3.x)
os.makedirs('mlruns', exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

experiment_name = "predictor-precios-california"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
except:
    experiment_id = mlflow.create_experiment(experiment_name)

print(f"📊 Experimento: {experiment_name}")
print(f"🔍 Experiment ID: {experiment_id}\n")

# ============================================
# CARGAR Y PREPARAR DATOS
# ============================================

print("📥 Cargando datos...")
datos = fetch_california_housing()
X = datos.data
y = datos.target

print(f"✅ Datos cargados: {X.shape[0]} casas, {X.shape[1]} features\n")

scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_normalizado, y,
    test_size=0.2,
    random_state=42
)

print(f"✅ Datos preparados:")
print(f"   Entrenamiento: {X_train.shape[0]} muestras")
print(f"   Prueba: {X_test.shape[0]} muestras\n")

# ============================================
# CREAR MODELO Y ENTRENAR CON MLFLOW
# ============================================

with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="v1.2.0-optimizado"
):

    print("🚀 Iniciando entrenamiento...\n")

    # Parámetros
    epocas = 500
    batch_size = 16
    dropout_rate = 0.4
    lr_rate = 0.0008
    arquitectura = "128-64-32-16"

    mlflow.log_params({
        'epocas': epocas,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'learning_rate': lr_rate,
        'arquitectura': arquitectura,
        'dataset': 'California Housing (1990)',
        'normalizacion': 'StandardScaler'
    })

    print("✅ Parámetros registrados en MLflow\n")

    # Crear modelo
    modelo = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(8,)),
        layers.Dropout(dropout_rate),

        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(16, activation='relu'),

        layers.Dense(1)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=lr_rate)
    modelo.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Early Stopping: parar cuando validación no mejore
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_mae',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    # Entrenar
    print("🔄 Entrenando modelo...")
    historial = modelo.fit(
        X_train, y_train,
        epochs=epocas,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    print("\n✅ Entrenamiento completado\n")

    # Evaluar
    print("📊 Evaluando en datos de prueba...")
    loss, mae = modelo.evaluate(X_test, y_test, verbose=0)

    mlflow.log_metrics({
        'test_loss': float(loss),
        'test_mae': float(mae),
        'test_mae_dollars': float(mae * 100000)
    })

    # Log histórico
    for epoch, (train_loss, val_loss, train_mae, val_mae) in enumerate(
        zip(historial.history['loss'],
            historial.history['val_loss'],
            historial.history['mae'],
            historial.history['val_mae'])
    ):
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mae': train_mae,
            'val_mae': val_mae
        }, step=epoch)

    print(f"✅ MAE en prueba: ${mae*100000:.2f}")
    print(f"   Error promedio: {mae*100:.2f}%\n")

    # Guardar artefactos
    print("💾 Guardando artefactos...")

    os.makedirs('artifacts_temp', exist_ok=True)

    with open('artifacts_temp/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact('artifacts_temp/scaler.pkl')

    info_modelo = {
        'version': 'v1.2.0',
        'descripcion': 'Modelo optimizado: Early Stopping + Dropout 0.4 + LR 0.0008 + Batch 16',
        'metricas': {
            'mae': float(mae * 100000),
            'mae_porcentaje': float(mae * 100),
            'test_loss': float(loss)
        },
        'arquitectura': {
            'capas': [128, 64, 32, 16],
            'dropout': dropout_rate,
            'epocas': epocas,
            'batch_size': batch_size
        },
        'dataset': 'California Housing (1990)',
        'features': list(datos.feature_names)
    }

    with open('artifacts_temp/info_modelo.json', 'w') as f:
        json.dump(info_modelo, f, indent=2)
    mlflow.log_artifact('artifacts_temp/info_modelo.json')

    mlflow.keras.log_model(
        modelo,
        artifact_path='modelo',
        registered_model_name='predictor-precios'
    )

    print("✅ Modelo registrado en MLflow Registry\n")

    # Resumen
    print("=" * 50)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 50)
    print(f"Versión: v1.2.0")
    print(f"MAE: ${mae*100000:.2f}")
    print(f"Parámetros: {epocas} épocas, arquitectura {arquitectura}")
    print("=" * 50)
    print("\n📊 Ver resultados:")
    print("   Ejecuta: mlflow ui")
    print("   Abre: http://localhost:5000")
    print("\n")

print("🎉 ¡Entrenamiento completado! Ahora ejecuta: mlflow ui")
