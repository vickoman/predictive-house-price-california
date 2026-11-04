#!/bin/bash

# Script para configurar y ejecutar el proyecto

echo "🚀 Predictor de Precios de Casas - Setup"
echo "=========================================="
echo ""

# Verificar si .venv existe
if [ ! -d ".venv" ]; then
    echo "📦 Creando virtual environment..."
    python3.12 -m venv .venv
fi

# Activar venv
echo "🔌 Activando virtual environment..."
source .venv/bin/activate

# Crear carpetas
echo "📁 Creando estructura de carpetas..."
mkdir -p src notebooks modelos datos mlruns

echo ""
echo "✅ Setup completado"
echo ""
echo "Próximos pasos:"
echo "==============="
echo ""
echo "1. Entrenar modelo:"
echo "   python src/entrenar.py"
echo ""
echo "2. Ver resultados (en otra terminal):"
echo "   mlflow ui"
echo "   # Abre http://localhost:5000"
echo ""
echo "3. Ejecutar API:"
echo "   python src/app.py"
echo ""
echo "4. Probar predicción:"
echo "   curl -X POST http://localhost:5000/predecir \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"medinc\":8,\"houseage\":20,\"averrooms\":6,\"avebedrooms\":1,\"population\":1000,\"aveoccup\":3,\"latitude\":37.8,\"longitude\":-122.2}'"
echo ""
