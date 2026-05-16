# ── Django ───────────────────────────────────────────────────────────────
from django.shortcuts import render

# ── Librerías de la Red Neuronal ─────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Librerías para gráficas ──────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# ── Librerías de ML ──────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)
# ── TensorFlow ───────────────────────────────────────────────────────────
import tensorflow as tf

# ── Preprocesamiento de texto ────────────────────────────────────────────
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Descarga de recursos NLTK (solo primera vez)
nltk.download('stopwords', quiet=True)

# Configuración NLP
_STOPWORDS_ES = set(stopwords.words('spanish'))
_STEMMER_ES = SnowballStemmer('spanish')

# ── Manejo de archivos ───────────────────────────────────────────────────
import os


# ─────────────────────────────────────────────────────────────────────────
# Función de limpieza de texto
# ─────────────────────────────────────────────────────────────────────────
def limpiar_texto(texto: str) -> str:
    """
    Pipeline de preprocesamiento de texto:
    Texto Crudo → Limpieza → Normalización → Tokenización →
    Stopwords → Stemming → Texto Procesado
    """

    # 1. LIMPIEZA
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    texto = re.sub(r'\S+@\S+\.\S+', '', texto)
    texto = re.sub(r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]', '', texto)
    texto = texto.lower()
    texto = re.sub(r'\s+', ' ', texto).strip()

    # 2. NORMALIZACIÓN
    texto_normalizado = unicodedata.normalize('NFD', texto)
    texto = ''.join(
        c for c in texto_normalizado
        if unicodedata.category(c) != 'Mn'
    )

    # 3. TOKENIZACIÓN
    tokens = texto.split()

    # 4. STOPWORDS
    tokens = [t for t in tokens if t not in _STOPWORDS_ES]

    # 5. STEMMING
    tokens = [_STEMMER_ES.stem(t) for t in tokens]

    # 6. TEXTO FINAL
    return ' '.join(tokens)


# ── Variables globales ───────────────────────────────────────────────────
_tokenizer_global = None
_label_encoder = None


def get_tokenizer_vocab():
    global _tokenizer_global
    if _tokenizer_global:
        return len(_tokenizer_global.word_index)
    return 0


def get_sequences(texts, tokenizer, train=True, max_seq_len=None):
    sequence = tokenizer.texts_to_sequences(texts)

    if train:
        max_seq_len = np.max(list(map(len, sequence)))

    sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=max_seq_len,
        padding='post'
    )
    return sequence


# ─────────────────────────────────────────────────────────────────────────
# Preprocesamiento completo
# ─────────────────────────────────────────────────────────────────────────
def preprocessing(df):
    """Preprocesa el DataFrame y devuelve train/test."""

    global _tokenizer_global, _label_encoder

    df = df.copy()

    print("Aplicando pipeline de texto...")
    df['SINTOMAS'] = df['SINTOMAS'].apply(limpiar_texto)
    print("Pipeline completado.")

    X = df['SINTOMAS']
    Y = df['ENFERMEDAD']

    # Codificación de etiquetas
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    _label_encoder = label_encoder

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y_encoded,
        test_size=0.30,
        shuffle=True,
        random_state=1
    )

    # Tokenización
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x_train)
    _tokenizer_global = tokenizer

    x_train = get_sequences(x_train, tokenizer, train=True)
    x_test = get_sequences(
        x_test,
        tokenizer,
        train=False,
        max_seq_len=x_train.shape[1]
    )

    print(f'Vocabulario: {len(tokenizer.word_index) + 1}')
    print(f'Secuencia: {x_train.shape[1]}')
    print(f'Clases: {len(label_encoder.classes_)}')
    print(f'Enfermedades: {list(label_encoder.classes_)}')

    return x_train, x_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────
# Vistas Django
# ─────────────────────────────────────────────────────────────────────────
def main(request):
    """Vista principal."""
    return render(request, 'index.html', context={})


def prediccion(request):
    """Entrena el modelo y devuelve métricas + primeras 50 predicciones."""

    # Ruta dataset
    base_path = os.path.expanduser(
        "~/Documentos/PROJECTS/IA/SIDM — Sistema Inteligente de Diagnóstico Médico/django_deep_learning/proyecto/dataset/"
    )

    data = pd.read_csv(f"{base_path}/ENFERMEDADES_SINTOMAS.csv")
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Preprocesamiento
    x_train, x_test, y_train, y_test = preprocessing(data)

    # ── Modelo ────────────────────────────────────────────────────────────
    vocab_size = get_tokenizer_vocab() + 1
    num_classes = len(np.unique(y_train))

    inputs = tf.keras.Input(shape=(x_train.shape[1],))
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=300)(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    # ── Entrenamiento ────────────────────────────────────────────────────
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # ── Evaluación ────────────────────────────────────────────────────────
    y_prob = model.predict(x_test, verbose=0)          # probabilidades por clase
    y_pred = np.argmax(y_prob, axis=1)                 # clase predicha

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    top3_acc = top_k_accuracy_score(y_test, y_prob, k=3, labels=np.arange(num_classes))

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    # ── Nombres de clases ────────────────────────────────────────────────
    label_encoder = _label_encoder
    class_names = list(label_encoder.classes_)

    # ── Primeros 50 registros ─────────────────────────────────────────────
    prediccion_50 = []
    n = min(50, len(y_pred))

    for i in range(n):
        real_idx = int(y_test[i])
        pred_idx = int(y_pred[i])

        top3_idx = np.argsort(y_prob[i])[::-1][:3]
        top3 = [
            {
                "enfermedad": class_names[idx],
                "probabilidad": float(y_prob[i][idx])
            }
            for idx in top3_idx
        ]

        prediccion_50.append({
            "id": i + 1,
            "real": class_names[real_idx],
            "predicho": class_names[pred_idx],
            "confianza": float(np.max(y_prob[i])),
            "acierto": "Sí" if real_idx == pred_idx else "No",
            "top3": top3
        })

    # ── Métricas finales ──────────────────────────────────────────────────
    metricas = {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bacc), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "loss_train_final": round(float(history.history["loss"][-1]), 4),
        "accuracy_train_final": round(float(history.history["accuracy"][-1]), 4),
        "val_loss_final": round(float(history.history["val_loss"][-1]), 4),
        "val_accuracy_final": round(float(history.history["val_accuracy"][-1]), 4),
    }

    context = {
        "metricas": metricas,
        "matriz_confusion": cm.tolist(),
        "reporte_clasificacion": report,
        "prediccion": prediccion_50,
        "clases": class_names,
    }

    return render(request, 'index.html', context=context)