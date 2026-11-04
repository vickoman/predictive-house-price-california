# 🏠 Predictor de Precios de Casas - California

Red neuronal entrenada con MLflow para predecir precios de casas en California basado en características de la propiedad.

## 📊 Características del Modelo

- **Arquitectura**: Red neuronal con 4 capas densas (128-64-32-16)
- **Precisión**: Error promedio ±$35,749 (7%)
- **Datos**: California Housing Dataset (1990, 20,640 casas)
- **Framework**: TensorFlow 2.16.2 + MLflow
- **Versionado**: Automático con MLflow Registry

## 🚀 Inicio Rápido

### 1. Activar environment

```bash
source .venv/bin/activate
```

### 2. Entrenar modelo

```bash
python src/entrenar.py
```

Esto creará una versión nueva del modelo en MLflow.

### 3. Ver MLflow UI

En otra terminal:

```bash
mlflow ui
```

Abre: http://localhost:5000

### 4. Ejecutar API

```bash
python src/app.py
```

API disponible en: http://localhost:5000

## 📡 Usar la API

### Health check

```bash
curl http://localhost:5000/health
```

### Información del modelo

```bash
curl http://localhost:5000/info
```

### Predicción individual

```bash
curl -X POST http://localhost:5000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "medinc": 8.0,
    "houseage": 20,
    "averrooms": 6,
    "avebedrooms": 1,
    "population": 1000,
    "aveoccup": 3,
    "latitude": 37.8,
    "longitude": -122.2
  }'
```

**Respuesta:**

```json
{
  "precio": 487200.50,
  "error_estimado": 35749.29,
  "rango": {
    "minimo": 451451.21,
    "maximo": 522949.79
  }
}
```

### Predicción en lote

```bash
curl -X POST http://localhost:5000/lote \
  -H "Content-Type: application/json" \
  -d '[
    {"medinc": 8.0, "houseage": 20, "averrooms": 6, "avebedrooms": 1, "population": 1000, "aveoccup": 3, "latitude": 37.8, "longitude": -122.2},
    {"medinc": 6.0, "houseage": 15, "averrooms": 5, "avebedrooms": 1, "population": 800, "aveoccup": 2.5, "latitude": 38.0, "longitude": -122.0}
  ]'
```

## 📁 Estructura del Proyecto

```
predictive-house-price-in-california/
├── .venv/                      # Virtual environment
├── src/
│   ├── entrenar.py            # Script de entrenamiento
│   ├── app.py                 # API Flask
│   └── predictor_mlflow.py    # Cargador de modelos
├── modelos/                    # Modelos locales (backup)
├── datos/                      # Datos de entrada
├── notebooks/                  # Análisis exploratorio
├── mlruns/                     # MLflow tracking (auto-generado)
├── requirements.txt            # Dependencias
├── .gitignore
└── README.md
```

## 🔧 Características de Entrada

| Feature | Descripción | Rango |
|---------|-------------|-------|
| `medinc` | Ingreso medio del bloque | 0.5 - 15.0 |
| `houseage` | Años de antigüedad | 1 - 52 |
| `averrooms` | Habitaciones promedio | 1.0 - 141.9 |
| `avebedrooms` | Dormitorios promedio | 0.4 - 34.4 |
| `population` | Población del bloque | 3 - 35,682 |
| `aveoccup` | Ocupantes promedio | 0.7 - 55.0 |
| `latitude` | Latitud | 32.5 - 42.0 |
| `longitude` | Longitud | -124.5 - -114.0 |

## 📈 Métricas

- **MAE (Error Absoluto Medio)**: $35,749
- **Error Porcentual**: 7.15%
- **Diferencia Train-Val**: 0.036 (sin overfitting)
- **Muestras de entrenamiento**: 16,512
- **Muestras de prueba**: 4,128

## ⚠️ Limitaciones

- ⚠️ Modelo entrenado SOLO para California
- ⚠️ Datos de 1990 (precios pueden haber cambiado)
- ⚠️ Validez dentro de rango: Lat [32.5, 42.0], Lon [-124.5, -114.0]

## 🔄 Workflow de Versionado

### Ver todas las versiones

```bash
mlflow ui
# Abre http://localhost:5000
# Ve a Models → predictor-precios
```

### Entrenar versión nueva

Modifica `src/entrenar.py`:
```python
with mlflow.start_run(run_name="v1.1.0-mejorado"):
    # Cambios...
```

```bash
python src/entrenar.py
```

### Usar versión específica en API

Modificar `src/app.py`:
```python
predictor = PredictorMLflow(version='2')  # o 'latest'
```

## 🐳 Docker

```bash
docker build -t predictor-precios .
docker run -p 5000:5000 predictor-precios
```

## 📚 Dependencias

- **TensorFlow 2.16.2**: Deep learning
- **scikit-learn**: Preprocessing y utilidades
- **MLflow 2.10.0**: Versionado y tracking
- **Flask 2.3.0**: API REST
- **Pandas/NumPy**: Data manipulation

## 👨‍💻 Desarrollo

### Crear entorno

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Entrenar

```bash
python src/entrenar.py
```

### Probar API

```bash
python src/app.py
# En otra terminal:
curl -X POST http://localhost:5000/predecir -H "Content-Type: application/json" -d '{"medinc":8,"houseage":20,"averrooms":6,"avebedrooms":1,"population":1000,"aveoccup":3,"latitude":37.8,"longitude":-122.2}'
```

### Ver resultados

```bash
mlflow ui
# http://localhost:5000
```

## 📝 Licencia

MIT

## 👤 Autor

Creado como ejemplo educativo de MLOps con Python.

---

**¿Preguntas?** Revisa los scripts en `src/` para entender cómo funciona todo.
