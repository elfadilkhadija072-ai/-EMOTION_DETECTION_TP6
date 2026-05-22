"""
emotionDetection.py
TP6 - Apprentissage Automatique (Partie 2)
Système de Détection des Émotions - CNN + Fine Tuning (Version Spéciale Dossiers)
"""

# ============================================================
# ETAPE 1 : IMPORTATION DES BIBLIOTHEQUES (SANS PANDAS)
# ============================================================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from tensorflow.keras import layers, models
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Configuration des constantes - ICI ON UTILISED DIRECTEMENT VOS DOSSIERS
IMG_SIZE = 48        
NUM_CLASSES = 7     
TRAIN_DIR = "train" 
TEST_DIR = "test"   

# Liste officielle des sous-dossiers extraits de votre ZIP
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_data_from_directory(directory_path):
    images = []
    labels = []

    if not os.path.exists(directory_path):
        raise Exception(f"Dossier introuvable : {directory_path}")

    print(f"[INFO] Chargement depuis : {directory_path}")

    for idx, emotion in enumerate(emotion_labels):
        folder = os.path.join(directory_path, emotion)

        if not os.path.exists(folder):
            print(f"[WARNING] Dossier manquant : {folder}")
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (48, 48))

            images.append(img)
            labels.append(idx)

    return np.array(images), np.array(labels)
# Chargement physique des images de vos dossiers
X_train_raw, y_train_raw = load_data_from_directory(TRAIN_DIR)
X_test_raw, y_test_raw = load_data_from_directory(TEST_DIR)

# Normalisation et Redimensionnement pour le CNN
X_train = X_train_raw.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
X_test  = X_test_raw.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

y_train = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
y_test  = to_categorical(y_test_raw, num_classes=NUM_CLASSES)

# Création de l'échantillon de validation réclamé par le TP
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

print(f"\n[SUCCÈS] Toutes les images ont été chargées sans aucun fichier CSV !")
print(f"       - Train : {X_train.shape[0]} | Val : {X_val.shape[0]} | Test : {X_test.shape[0]}\n")


# ============================================================
# ETAPE 3 : ARCHITECTURE DU MODELE CNN
# ============================================================
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp['neurons'], (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp['neurons'] * 2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(hp['neurons'] * 2, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# ETAPE 4 : TUNING ET RECHERCHE D'HYPERPARAMETRES
# ============================================================
def hyperparameter_search(X_train, y_train, X_val, y_val, num_random=2):
    param_grid = {
        'neurons': [32, 64],
        'num_layers': [2],
        'lr': [0.001, 0.0005],
        'batch_size': [64]
    }
    
    grid = list(ParameterGrid(param_grid))
    random_samples = random.sample(grid, min(num_random, len(grid)))
    
    best_accuracy = -1
    best_hp = None
    best_model = None
    
    print(f"[TUNING] Test de {len(random_samples)} combinaisons pour votre PC...")
    
    for idx, hp in enumerate(random_samples):
        print(f" -> Essai {idx+1}/{len(random_samples)} | Paramètres : {hp}")
        model = build_model(hp)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=2,
            batch_size=hp['batch_size'],
            verbose=1
        )
        
        val_acc = history.history['val_accuracy'][-1]
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_hp = hp
            best_model = model
            
        tf.keras.backend.clear_session() # Libère votre mémoire vive
            
    return best_model, best_hp

best_model, best_hyperparams = hyperparameter_search(X_train, y_train, X_val, y_val, num_random=2)


# ============================================================
# ETAPE 5 : ENTRAÎNEMENT FINAL ET SAUVEGARDE DU MODELE .H5
# ============================================================
print("\n[INFO] Entraînement final...")
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5, 
    batch_size=best_hyperparams['batch_size'],
    verbose=1
)

best_model.save("best_fer2013_emotion_model.h5")
best_model.save("fer2013_emotion_model.h5")
print("[INFO] Fichiers modèles .h5 sauvegardés avec succès.")


# ============================================================
# ETAPE 6 : ETAPES REPRISES DU TP (METRIQUES ET COURBES ROC)
# ============================================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.savefig("training_metrics.png")
plt.close()

# Évaluation finale sur l'ensemble Test
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

display_labels = [l.capitalize() for l in emotion_labels]
print("\n--- RAPPORT DE CLASSIFICATION ---")
print(classification_report(y_test_classes, y_pred_classes, target_names=display_labels))

# Courbes ROC finales
y_test_bin = label_binarize(y_test_classes, classes=list(range(NUM_CLASSES)))
plt.figure(figsize=(8, 6))
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    plt.plot(fpr, tpr, label=f"{display_labels[i]} (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Courbes ROC')
plt.legend()
plt.savefig("roc_curves.png")
plt.close()

print("\n============================================================")
print("TOUT EST FINI ENTIÈREMENT ET AVEC SUCCÈS !")
print("============================================================")