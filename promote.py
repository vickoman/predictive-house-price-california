#!/usr/bin/env python3
"""
Promover v1.2.0 (Version 3) a Production
"""

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
client = MlflowClient()

print("🚀 Promoviendo predictor-precios v1.2.0 a Production...\n")

try:
    # En MLflow 3.x, usar transition_model_version_stage
    client.transition_model_version_stage(
        name="predictor-precios",
        version="3",
        stage="Production"
    )
    print("✅ EXITOSO: v1.2.0 (Version 3) promovido a Production\n")

    # Mostrar versiones
    versions = client.search_model_versions("name='predictor-precios'")
    print("📊 Estado actual de versiones:\n")
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"  Version {v.version}: Stage = {v.current_stage}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
