# ============================================================
#  predict_external_images.py
#  Prédiction d'émotions sur :
#    1. des images du dossier test/ Kaggle
#    2. vos propres images externes (test.jpeg, test2.jpeg, …)
#
#  Prérequis : avoir exécuté emotionDetection.py au moins une fois
#              pour générer best_fer2013_emotion_model.h5
# ============================================================

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# ── Paramètres ───────────────────────────────────────────────────────────────
MODEL_PATH     = "best_fer2013_emotion_model.h5"   # ou fer2013_emotion_model.h5
TEST_DIR       = "test"                             # dossier test Kaggle
IMG_SIZE       = 48
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Images externes placées dans le même dossier que ce script
EXTERNAL_IMAGES = ["test.jpeg", "test2.jpeg"]      # ajoutez vos fichiers ici

# Nombre d'images Kaggle à afficher (tirées aléatoirement)
NB_KAGGLE_SAMPLES = 5

# ── Chargement du modèle ─────────────────────────────────────────────────────
print(f"[INFO] Chargement du modèle depuis : {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Modèle chargé avec succès !\n")

# ── Fonction de prétraitement ─────────────────────────────────────────────────
def preprocess(img_path, img_size=IMG_SIZE):
    """Lit une image, la met en niveaux de gris, la redimensionne et la normalise."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    img_resized    = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_reshaped   = img_normalized.reshape(img_size, img_size, 1)
    return img_resized, img_reshaped   # (brute pour affichage, normalisée pour le modèle)

# ── Fonction de prédiction + affichage ───────────────────────────────────────
def predict_and_display(image_paths, title="Emotion Predictions"):
    """Prétraite une liste de chemins d'images, prédit et affiche les résultats."""
    raw_images    = []
    preprocessed  = []
    valid_paths   = []

    for path in image_paths:
        try:
            raw, processed = preprocess(path)
            raw_images.append(raw)
            preprocessed.append(processed)
            valid_paths.append(path)
        except FileNotFoundError as e:
            print(f"[AVERTISSEMENT] {e}")

    if not valid_paths:
        print("[ERREUR] Aucune image valide trouvée.")
        return

    images_np  = np.array(preprocessed)
    predictions = model.predict(images_np, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    confidences  = np.max(predictions, axis=1)

    n = len(valid_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(raw_images[i], cmap='gray')
        emotion    = EMOTION_LABELS[pred_classes[i]]
        confidence = confidences[i]
        ax.set_title(f"Prédit : {emotion}\n({confidence:.0%})", fontsize=10)
        ax.axis("off")

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# ── 1. Prédiction sur images du dossier test/ Kaggle ─────────────────────────
print("=" * 60)
print("  PARTIE 1 : Images Kaggle (test/)")
print("=" * 60)

kaggle_paths = []
for emotion in EMOTION_LABELS:
    folder = os.path.join(TEST_DIR, emotion)
    if not os.path.isdir(folder):
        continue
    files = [os.path.join(folder, f)
             for f in os.listdir(folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if files:
        kaggle_paths.append(random.choice(files))  # 1 image par classe

# Prendre au max NB_KAGGLE_SAMPLES images
random.shuffle(kaggle_paths)
kaggle_paths = kaggle_paths[:NB_KAGGLE_SAMPLES]

if kaggle_paths:
    predict_and_display(kaggle_paths, title="Kaggle Test Set Predictions")
else:
    print(f"[AVERTISSEMENT] Dossier '{TEST_DIR}' introuvable ou vide.")

# ── 2. Prédiction sur vos images externes ────────────────────────────────────
print("\n" + "=" * 60)
print("  PARTIE 2 : Images externes")
print("=" * 60)

external_existing = [p for p in EXTERNAL_IMAGES if os.path.exists(p)]
if external_existing:
    predict_and_display(external_existing, title="External Images Predictions")
else:
    print("[INFO] Aucune image externe trouvée dans la liste EXTERNAL_IMAGES.")
    print("       Ajoutez vos fichiers dans la variable EXTERNAL_IMAGES en haut du script.")

# ── 3. Matrice de probabilités pour chaque image ─────────────────────────────
print("\n" + "=" * 60)
print("  PARTIE 3 : Détail des probabilités par émotion")
print("=" * 60)

all_paths = kaggle_paths + external_existing
if all_paths:
    _, preprocessed = zip(*[preprocess(p) for p in all_paths])
    images_np = np.array(preprocessed)
    preds = model.predict(images_np, verbose=0)

    for i, path in enumerate(all_paths):
        print(f"\n  Image : {os.path.basename(path)}")
        for j, label in enumerate(EMOTION_LABELS):
            bar = "█" * int(preds[i][j] * 30)
            print(f"    {label:10s} : {preds[i][j]:.3f}  {bar}")

print("\n[INFO] Terminé.")
