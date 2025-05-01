# 🔍 Analyseur de Réputation des Entreprises

## 🎯 Objectif

Ce projet vise à analyser la **réputation de l'entreprise Quicken** en construisant un modèle capable de classifier les **avis clients** en **avis positifs** ou **négatifs**.

Le processus comprend :  
1. 🔄 La récupération automatique d'avis depuis **TrustPilot**  
2. 🧠 Le *fine-tuning* d'un modèle **BERT** pour la classification  
3. 📊 L’évaluation des performances du modèle  
4. 🌐 La création d'une **interface web** pour l'inférence  

## 🗂️ Structure du Projet

- `reviews_scrapper.py` 🕷️ : Script de **scraping** des avis clients depuis TrustPilot  
- `quicken_reviews.csv` 📄 : Jeu de données contenant les avis collectés et labellisés  
- `training.ipynb` 📘 : Notebook pour le *fine-tuning* de **BERT**  
- `app.py` 🖥️ : Interface web permettant à l'utilisateur de saisir un avis et d'obtenir la prédiction en temps réel  

## 🛠️ Technologies Utilisées

- 🐍 Python  
- 🤗 Hugging Face Transformers (**BERT**)  
- 🔥 PyTorch  
- 📈 Scikit-learn (**accuracy**, **F1-score**)  
- 🌐 BeautifulSoup & Requests (**scraping web**)  
- 📊 Pandas, NumPy  
- 🎨 Streamlit (**interface web** pour l'inférence)  

## ✨ Fonctionnalités

- ✅❌ **Classification binaire** des avis (positif/négatif)  
- 📉 **Visualisation des statistiques** d'analyse  
- 🕓 **Historique des analyses récentes**  
- ⚡ **Exemples d’avis prédéfinis** pour tester rapidement l'application  

## ⚙️ Installation

Assurez-vous d'avoir installé les dépendances :

```bash
pip install -r requirements.txt
```

## 🚀 Lancer l'Application

Exécutez l'application Streamlit :

```bash
streamlit run app.py
```

## 🎥 Demo Vidéo
https://github.com/user-attachments/assets/c5e7be03-7e32-437b-9d02-a71ecad819f2



