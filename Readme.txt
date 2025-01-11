# Système de Gestion et Visualisation de Données

Ce projet est une application web Django permettant d'analyser et de visualiser des données à partir de fichiers CSV.

## Fonctionnalités

- Upload de fichiers CSV
- Analyse univariée et bivariée des données
- Visualisations variées avec Matplotlib et Seaborn
- Interface utilisateur interactive
- Calculs statistiques complets

## Installation

1. Cloner le repository :
```bash
git clone <url-du-repo>
cd data-analysis-project
```

2. Créer un environnement virtuel et l'activer :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
venv\Scripts\activate     # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Effectuer les migrations :
```bash
python manage.py makemigrations
python manage.py migrate
```

5. Lancer le serveur :
```bash
python manage.py runserver
```

## Utilisation

1. Accédez à l'application via `http://localhost:8000`
2. Téléchargez un fichier CSV
