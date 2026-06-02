# QDA Backend API

API backend pour l'analyse discriminante quadratique (QDA). Permet d'uploader des fichiers CSV, d'entraîner des modèles de classification et d'obtenir des prédictions.

## Stack technique

- **Framework** : FastAPI
- **ML classique** : scikit-learn (QuadraticDiscriminantAnalysis)
- **ML neuronal** : PyTorch (réseau de neurones fully connected)
- **Données** : pandas, numpy

## Fonctionnalités

Deux variantes de modèle QDA sont disponibles :

1. **Sklearn QDA** — implémentation classique via scikit-learn avec standardisation des données
2. **Neural QDA** — réseau de neurones (2 couches cachées ReLU) entraîné pour la classification

Les données sont uploadées au format CSV. L'entraînement se fait automatiquement si le fichier contient une colonne cible. Les résultats de prédiction sont stockés en mémoire et récupérables via un identifiant unique.

## API exposées

Toutes les routes sont préfixées par `/qda`.

### Sklearn

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | `/qda/sklearn/train` | Entraîner le modèle sklearn QDA |
| POST | `/qda/sklearn/predict` | Prédire avec le modèle sklearn |
| GET | `/qda/sklearn/model-info` | Infos sur le modèle sklearn entraîné |

### Neural

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | `/qda/neural/train` | Entraîner le réseau de neurones |
| POST | `/qda/neural/predict` | Prédire avec le réseau de neurones |
| GET | `/qda/neural/model-info` | Infos sur le modèle neural entraîné |

### Utilitaires

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/qda/result/{result_id}` | Récupérer un résultat de prédiction |

## Lancement

```bash
uvicorn main:app --reload
```

Documentation interactive disponible sur `/docs` (Swagger) et `/redoc` (ReDoc).


### Docker

```bash
docker build -t qda-backend .
docker run -p 8000:8000 --env-file .env qda-backend
```

# 1. Créer les fichiers ci-dessus
# 2. Installer pydantic-settings
pip install pydantic-settings
pip freeze > requirements.txt

# 3. Tester que tout fonctionne
python3 main.py

# 4. Commiter
git add .env.example config.py install.sh Dockerfile .dockerignore README.md
git add main.py services/qda_service.py services/neural_service.py
git add requirements.txt .gitignore
git commit -m "devops: config centralisée, Docker, install script"
git push