# 🎭 Détection des Émotions par Deep Learning (CNN)

Ce projet implémente un pipeline complet de reconnaissance d'émotions faciales en utilisant un réseau de neurones convolutionnel (CNN) développé avec **TensorFlow / Keras**. Il intègre également une couche d'interprétabilité avec **SHAP** et s'inscrit dans une démarche **DevOps** grâce à un versionning et un suivi rigoureux avec Git.

Le système est entraîné sur le dataset **FER-2013** (Kaggle) et classifie les visages selon 7 émotions : *Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*.

---

## 📌 Objectifs Pédagogiques
* **Initiation aux réseaux de neurones** : Conception, structuration et optimisation d'une architecture CNN.
* **Computer Vision (Perception)** : Utilisation du traitement d'images comme capteur perceptif reliant le système intelligent à son environnement physique.
* **Interprétabilité algorithmique** : Analyse de la contribution des pixels aux décisions du modèle à l'aide des valeurs SHAP.
* **Fine-Tuning avancé** : Recherche d'hyperparamètres par Random Search et Grid Search.
* **Évaluation Métrique** : Analyse via Matrice de confusion, courbes ROC, F1-Score et F2-Score.

---

## 🛠️ Architecture du Pipeline & Modèle

Le modèle suit une architecture séquentielle classique et performante pour la vision par ordinateur :

| Couche | Paramètres | Rôle |
| :--- | :--- | :--- |
| **Conv2D (×3)** | 32/64/128 filtres, noyau 3×3, ReLU | Extraction des caractéristiques visuelles (contours, textures) |
| **MaxPooling2D** | Fenêtre 2×2 | Réduction de dimension spatiale et invariance |
| **Dropout** | Taux configurable (0.3 – 0.5) | Régularisation et prévention du surapprentissage |
| **Flatten** | — | Conversion du tenseur 3D en vecteur 1D |
| **Dense** | 128 neurones, ReLU | Apprentissage des relations complexes entre features |
| **Dense (Sortie)** | 7 neurones, Softmax | Probabilités finales pour chacune des 7 émotions |

---

## 📈 Métriques d'Entraînement et Performance

Le script principal génère un suivi complet de la phase d'apprentissage ainsi que les cartes d'explication de pixels :

```bash
python emotionDetection.py