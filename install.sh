#!/bin/bash
set -e

echo "📦 Installation des dépendances..."
pip install -r requirements.txt

echo "🔍 Détection GPU/CPU..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✅ GPU détecté — torch déjà compatible"
else
    echo "💻 CPU uniquement — installation torch CPU"
    pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "✅ Environnement prêt !"
echo "👉 Lancez : python3 main.py"