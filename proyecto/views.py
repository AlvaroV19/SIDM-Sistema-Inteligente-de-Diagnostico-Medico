# ══════════════════════════════════════════════════════════════════════════════
# views.py — SIDM: Sistema Inteligente de Diagnóstico Médico
# Modelo: BETO (dccuchile/bert-base-spanish-wwm-cased)
#
# Arquitectura:
#   Texto crudo
#     → Limpieza mínima (URLs, emails, espacios)
#     → Tokenizador WordPiece de BETO (maneja acentos, puntuación, morfología)
#     → TFBertForSequenceClassification (fine-tuning completo)
#     → Predicción de enfermedad
# ══════════════════════════════════════════════════════════════════════════════

# ── Django ────────────────────────────────────────────────────────────────────
from django.shortcuts import render

# ── Librerías base ────────────────────────────────────────────────────────────
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

# ── TensorFlow ────────────────────────────────────────────────────────────────
import tensorflow as tf

# ── HuggingFace Transformers ──────────────────────────────────────────────────
# BertTokenizerFast  → tokenizador WordPiece optimizado en Rust, más rápido
#                      que BertTokenizer y con las mismas salidas.
# TFBertForSequenceClassification → BETO preentrenado + cabezal Dense de
#                                   clasificación (pooler [CLS] → num_classes).
from transformers import BertTokenizerFast, TFBertForSequenceClassification

# ── ML / métricas ─────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

import joblib
import pickle
import json

# ══════════════════════════════════════════════════════════════════════════════
# Constantes de configuración
# ══════════════════════════════════════════════════════════════════════════════

# Checkpoint oficial de BETO cased en HuggingFace Hub.
# "cased" preserva mayúsculas y acentos (á, é, ñ…), lo que es esencial
# para descripciones médicas en español con puntuación correcta.
BETO_CHECKPOINT = "dccuchile/bert-base-spanish-wwm-cased"

# Longitud máxima de secuencia tras tokenización WordPiece.
# 128 tokens cubre la gran mayoría de descripciones de síntomas;
# aumentar a 256 si el dataset contiene textos más largos.
MAX_SEQ_LEN = 128




# ── Estado global ─────────────────────────────────────────────────────────────
# Se inicializan una vez y se reutilizan en peticiones subsiguientes
# para evitar recargar el tokenizador en cada request.
_tokenizer_beto: BertTokenizerFast | None = None
_label_encoder: LabelEncoder | None = None

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpeta models
MODEL_DIR = BASE_DIR / "models"

# Crear directorio si no existe
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "modelo_beto")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_beto")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — Limpieza de texto
# ══════════════════════════════════════════════════════════════════════════════

def normalizar_reporte(report: dict) -> dict:
    """Renombra claves con guion/espacio para que Django templates pueda accederlas."""
    nuevo = {}
    for k, v in report.items():
        nueva_k = k.replace("-", "_").replace(" ", "_")
        if isinstance(v, dict):
            nuevo[nueva_k] = {
                ik.replace("-", "_"): iv for ik, iv in v.items()
            }
        else:
            nuevo[nueva_k] = v
    return nuevo

def limpiar_texto(texto: str) -> str:
    """
    Limpieza mínima y no destructiva, diseñada para BETO cased.

    A diferencia del pipeline anterior (stemming + stopwords + NFD),
    aquí se conserva la mayor parte del texto original porque BETO:
      - Maneja acentos y ñ nativamente (modelo cased, vocabulario español).
      - Usa WordPiece para descomponer palabras en subpalabras, capturando
        morfología sin necesidad de stemming.
      - Aprende por sí solo qué tokens son relevantes mediante la atención.

    Solo se eliminan ruidos que BETO no puede aprovechar semánticamente:
    URLs, correos y espacios redundantes.
    """
    if pd.isna(texto):
        return ""

    texto = str(texto)

    # Eliminar URLs (http, https, www)
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)

    # Eliminar direcciones de correo electrónico
    texto = re.sub(r"\S+@\S+\.\S+", "", texto)

    # Colapsar espacios múltiples y limpiar extremos
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — Tokenización BETO
# ══════════════════════════════════════════════════════════════════════════════

def get_tokenizer() -> BertTokenizerFast:
    """
    Devuelve el tokenizador de BETO, cargándolo una sola vez (singleton).

    BertTokenizerFast aplica tokenización WordPiece:
      - Divide palabras desconocidas en subpalabras ("enfermedad" → ["enfer", "##medad"]).
      - Agrega tokens especiales: [CLS] al inicio y [SEP] al final.
      - Genera input_ids (índices de vocabulario) y attention_mask
        (1 = token real, 0 = padding).
    """
    global _tokenizer_beto

    if _tokenizer_beto is None:
        _tokenizer_beto = BertTokenizerFast.from_pretrained(BETO_CHECKPOINT)

    return _tokenizer_beto


def encode_texts(
    texts: pd.Series | list,
    tokenizer: BertTokenizerFast,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte una lista de textos en tensores de entrada para BETO.

    Parámetros
    ----------
    texts     : textos ya limpios
    tokenizer : instancia de BertTokenizerFast

    Retorna
    -------
    input_ids      : array (n_samples, MAX_SEQ_LEN) — índices de subpalabras
    attention_mask : array (n_samples, MAX_SEQ_LEN) — 1 real / 0 padding
    """
    encoding = tokenizer(
        list(texts),
        max_length=MAX_SEQ_LEN,
        padding="max_length",   # rellena con [PAD] hasta MAX_SEQ_LEN
        truncation=True,         # recorta si supera MAX_SEQ_LEN
        return_tensors="np",     # devuelve NumPy para compatibilidad con Keras
    )

    return encoding["input_ids"], encoding["attention_mask"]


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — Preprocesamiento completo
# ══════════════════════════════════════════════════════════════════════════════

def preprocessing(df: pd.DataFrame):
    """
    Pipeline completo: DataFrame → conjuntos de entrenamiento y prueba.

    Pasos:
      1. Limpieza mínima de texto (limpiar_texto).
      2. Codificación de etiquetas con LabelEncoder.
      3. Split estratificado 70/30.
      4. Tokenización con BETO (encode_texts).

    Retorna
    -------
    x_train_ids, x_train_masks : inputs de entrenamiento
    x_test_ids,  x_test_masks  : inputs de prueba
    y_train, y_test            : etiquetas codificadas
    """
    global _label_encoder

    df = df.copy()

    # ── Paso 1: limpieza de texto ─────────────────────────────────────────
    df["SINTOMAS"] = df["SINTOMAS"].apply(limpiar_texto)

    X = df["SINTOMAS"]
    Y = df["ENFERMEDAD"]

    # ── Paso 2: codificación de etiquetas ─────────────────────────────────
    # LabelEncoder asigna un entero único a cada nombre de enfermedad.
    # Las clases quedan ordenadas alfabéticamente (reproducible).
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    _label_encoder = label_encoder

    # ── Paso 3: split train / test ────────────────────────────────────────
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        X,
        Y_encoded,
        test_size=0.30,
        shuffle=True,
        stratify=Y_encoded,
        random_state=1,
    )

    # ── Paso 4: tokenización con BETO ─────────────────────────────────────
    tokenizer = get_tokenizer()

    x_train_ids, x_train_masks = encode_texts(x_train_raw, tokenizer)
    x_test_ids,  x_test_masks  = encode_texts(x_test_raw,  tokenizer)

    return (
        x_train_ids, x_train_masks,
        x_test_ids,  x_test_masks,
        y_train, y_test,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4 — Construcción del modelo
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int) -> TFBertForSequenceClassification:
    """
    Carga BETO preentrenado y agrega un cabezal de clasificación.

    Arquitectura interna de TFBertForSequenceClassification:
      - 12 capas Transformer (Multi-Head Self-Attention + FFN).
      - Pooler: toma el vector del token [CLS] y aplica una Dense(768, tanh).
      - Clasificador: Dense(num_classes) sobre el pooler output.

    El fine-tuning actualiza TODOS los pesos (BETO + clasificador),
    lo que permite al modelo adaptar las representaciones a síntomas médicos.

    Parámetros
    ----------
    num_classes : número de enfermedades únicas en el dataset
    """
    model = TFBertForSequenceClassification.from_pretrained(
        BETO_CHECKPOINT,
        num_labels=num_classes,
    )

    # Optimizador Adam con LR pequeño para no sobreescribir pesos preentrenados
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    # from_logits=True porque TFBertForSequenceClassification devuelve logits
    # (sin softmax), lo que es numéricamente más estable que aplicar softmax
    # y luego usar CrossEntropy estándar.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 5 — Predicción con top-K y confianza
# ══════════════════════════════════════════════════════════════════════════════

def predecir(
    model: TFBertForSequenceClassification,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera predicciones y probabilidades a partir de los logits del modelo.

    Como el modelo devuelve logits (sin normalizar), se aplica softmax
    para obtener distribuciones de probabilidad interpretables.

    Retorna
    -------
    y_pred  : array (n,)   — índice de clase con mayor probabilidad
    y_prob  : array (n, C) — distribución de probabilidad sobre todas las clases
    """
    # Convertir a int32 (tipos que espera el modelo)
    input_ids = tf.cast(input_ids, tf.int32)
    attention_mask = tf.cast(attention_mask, tf.int32)
    
    # Crear token_type_ids (0s para todos los tokens, ya que no tenemos pares de oraciones)
    token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)
    
    # Llamar al modelo con los inputs correctos
    outputs = model(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        training=False,
    )

    # Convertir logits a probabilidades
    y_prob = tf.nn.softmax(outputs.logits, axis=-1).numpy()
    y_pred = np.argmax(y_prob, axis=1)

    return y_pred, y_prob


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 5.5 — Guardar y cargar modelo
# ══════════════════════════════════════════════════════════════════════════════

def guardar_modelo(
    model: TFBertForSequenceClassification,
    tokenizer: BertTokenizerFast,
    label_encoder: LabelEncoder,
) -> None:
    """
    Guarda el modelo, tokenizador y label_encoder en disco.
    Usa los métodos nativos de Transformers para compatibilidad máxima.
    """
    # Guardar modelo usando el método de Transformers
    model.save_pretrained(MODEL_PATH)
    
    # Guardar tokenizador
    tokenizer.save_pretrained(TOKENIZER_PATH)
    
    # Guardar label_encoder
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    
    print(f"✓ Modelo guardado en: {MODEL_PATH}")
    print(f"✓ Tokenizador guardado en: {TOKENIZER_PATH}")
    print(f"✓ Label encoder guardado en: {LABEL_ENCODER_PATH}")


def cargar_modelo() -> tuple[TFBertForSequenceClassification, BertTokenizerFast, LabelEncoder] | None:
    """
    Carga el modelo, tokenizador y label_encoder desde disco.
    Usa los métodos nativos de Transformers para compatibilidad máxima.
    Retorna (model, tokenizer, label_encoder) o None si no existen.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        return None
    
    try:
        # Cargar modelo usando el método de Transformers
        model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
        
        # Cargar tokenizador
        tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
        
        # Cargar label_encoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        
        return model, tokenizer, label_encoder
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return None



# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 6 — Vistas Django
# ══════════════════════════════════════════════════════════════════════════════

def main(request):
    """
    Vista principal — menú para elegir entre entrenar o diagnosticar.
    """
    # Verificar si existe un modelo entrenado
    modelo_existe = os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH)
    
    metrics_existe = os.path.exists(METRICS_PATH)

    context = {
        "modelo_existe": modelo_existe,
        "metrics_existe": metrics_existe,
    }
    return render(request, "index.html", context=context)


def entrenar(request):
    """
    Vista para entrenar el modelo.
    
    Flujo completo:
      1. Carga y limpieza del dataset.
      2. Preprocesamiento y tokenización con BETO.
      3. Construcción y fine-tuning del modelo.
      4. Evaluación con múltiples métricas.
      5. Guardado del modelo.
      6. Renderizado de resultados.
    """
    global _label_encoder

    # ── 1. Carga del dataset ──────────────────────────────────────────────
    dataset_path = BASE_DIR / "dataset"

    csv = pd.read_csv(f"{dataset_path}/ENFERMEDADES_SINTOMAS.csv")
    csv_aux = pd.read_csv(f"{dataset_path}/ENFERMEDADES_SINTOMAS_redux.csv")
    data = pd.concat([csv, csv_aux], ignore_index=True)  # Duplicar para aumentar tamaño
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # ── 2. Preprocesamiento ───────────────────────────────────────────────
    (
        x_train_ids, x_train_masks,
        x_test_ids,  x_test_masks,
        y_train, y_test,
    ) = preprocessing(data)

    num_classes = len(np.unique(y_train))

    # ── 3. Construcción del modelo ────────────────────────────────────────
    model = build_model(num_classes)

    # ── 4. Entrenamiento (fine-tuning) ────────────────────────────────────
    history = model.fit(
        {"input_ids": x_train_ids, "attention_mask": x_train_masks},
        y_train,
        validation_split=0.2,
        batch_size=16,
        epochs=5,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    # ── 5. Evaluación ─────────────────────────────────────────────────────
    y_pred, y_prob = predecir(model, x_test_ids, x_test_masks)

    acc    = accuracy_score(y_test, y_pred)
    bacc   = balanced_accuracy_score(y_test, y_pred)
    top3   = top_k_accuracy_score(y_test, y_prob, k=3, labels=np.arange(num_classes))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report = normalizar_reporte(report)
    cm     = confusion_matrix(y_test, y_pred)

    # ── 6. Nombres de clases ──────────────────────────────────────────────
    class_names = list(_label_encoder.classes_)

    # ── 7. Matriz de confusión etiquetada ─────────────────────────────────
    matriz_confusion_etiquetada = [
        {
            "real": class_names[i],
            "valores": [
                {
                    "pred": class_names[j],
                    "valor": int(cm[i, j]),
                    "es_diagonal": i == j,
                }
                for j in range(len(class_names))
            ],
        }
        for i in range(len(class_names))
    ]

    # ── 8. Primeras 50 predicciones detalladas ────────────────────────────
    prediccion_50 = []
    n = min(50, len(y_pred))

    for i in range(n):
        real_idx = int(y_test[i])
        pred_idx = int(y_pred[i])

        top3_idx = np.argsort(y_prob[i])[::-1][:3]
        top3_detalle = [
            {
                "enfermedad": class_names[idx],
                "probabilidad": float(y_prob[i][idx]),
            }
            for idx in top3_idx
        ]

        prediccion_50.append(
            {
                "id": i + 1,
                "real": class_names[real_idx],
                "predicho": class_names[pred_idx],
                "confianza": float(np.max(y_prob[i])),
                "acierto": "Sí" if real_idx == pred_idx else "No",
                "top3": top3_detalle,
            }
        )

    # ── 9. Métricas consolidadas ──────────────────────────────────────────
    metricas = {
        "accuracy":            round(float(acc),              4),
        "balanced_accuracy":   round(float(bacc),             4),
        "precision_macro":     round(float(precision_macro),  4),
        "recall_macro":        round(float(recall_macro),     4),
        "f1_macro":            round(float(f1_macro),         4),
        "precision_weighted":  round(float(precision_w),      4),
        "recall_weighted":     round(float(recall_w),         4),
        "f1_weighted":         round(float(f1_weighted),      4),
        "top3_accuracy":       round(float(top3),             4),
        "loss_train_final":    round(float(history.history["loss"][-1]),         4),
        "accuracy_train_final":round(float(history.history["accuracy"][-1]),     4),
        "val_loss_final":      round(float(history.history["val_loss"][-1]),      4),
        "val_accuracy_final":  round(float(history.history["val_accuracy"][-1]), 4),
    }

    # ── 10. Guardar modelo y métricas ─────────────────────────────────────
    guardar_modelo(model, get_tokenizer(), _label_encoder)

    # Guardar paquete de métricas/contexto para visualización posterior
    metrics_bundle = {
        "metricas": metricas,
        "matriz_confusion": cm.tolist(),
        "matriz_confusion_etiquetada": matriz_confusion_etiquetada,
        "reporte_clasificacion": report,
        "prediccion": prediccion_50,
        "clases": class_names,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    try:
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_bundle, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving metrics file: {e}")
    context = {
        "metricas":                    metricas,
        "matriz_confusion":            cm.tolist(),
        "matriz_confusion_etiquetada": matriz_confusion_etiquetada,
        "reporte_clasificacion":       report,
        "prediccion":                  prediccion_50,
        "clases":                      class_names,
        "tipo":                        "entrenar",
    }

    return render(request, "index.html", context=context)


def diagnosticar(request):
    """
    Vista para hacer diagnósticos usando un modelo entrenado.
    
    Si es GET: muestra el formulario para ingresar síntomas.
    Si es POST: procesa los síntomas y muestra el diagnóstico.
    """
    # Cargar modelo entrenado
    resultado = cargar_modelo()
    
    if resultado is None:
        context = {
            "error": "No hay modelo entrenado. Primero debe entrenar el modelo.",
            "tipo": "diagnosticar",
        }
        return render(request, "resultado.html", context=context)
    
    model, tokenizer, label_encoder = resultado
    
    if request.method == "GET":
        # Mostrar formulario
        context = {
            "tipo": "formulario",
        }
        return render(request, "resultado.html", context=context)
    
    elif request.method == "POST":
        # Procesar síntomas
        sintomas_usuario = request.POST.get("sintomas", "").strip()
        
        if not sintomas_usuario:
            context = {
                "error": "Por favor ingresa síntomas para diagnosticar.",
                "tipo": "formulario",
            }
            return render(request, "resultado.html", context=context)
        
        # Limpiar y tokenizar
        sintomas_limpios = limpiar_texto(sintomas_usuario)
        input_ids, attention_mask = encode_texts([sintomas_limpios], tokenizer)
        
        # Predecir
        y_pred, y_prob = predecir(model, input_ids, attention_mask)
        
        # Obtener resultados
        pred_idx = int(y_pred[0])
        probs = y_prob[0]
        
        # Top-3 enfermedades
        top3_idx = np.argsort(probs)[::-1][:3]
        top_3 = [
            {
                "nombre": label_encoder.classes_[idx],
                "probabilidad": float(probs[idx]) * 100,
            }
            for idx in top3_idx[1:]  # Saltar el primero para no repetir
        ]
        
        context = {
            "tipo": "diagnostico",
            "sintomas_usuario": sintomas_usuario,
            "enfermedad_principal": label_encoder.classes_[pred_idx],
            "probabilidad_principal": float(probs[pred_idx]) * 100,
            "top_3": top_3,
        }
        
        return render(request, "resultado.html", context=context)


def mostrar_metricas(request):
    """Vista para mostrar métricas guardadas sin reentrenar."""
    if not os.path.exists(METRICS_PATH):
        context = {"error": "No hay métricas guardadas. Entrena el modelo primero."}
        return render(request, "metricas.html", context=context)

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_bundle = json.load(f)

        if "reporte_clasificacion" in metrics_bundle:
            metrics_bundle["reporte_clasificacion"] = normalizar_reporte(
                metrics_bundle["reporte_clasificacion"]
            )
    
    except Exception as e:
        context = {"error": f"Error al leer métricas: {e}"}
        return render(request, "metricas.html", context=context)

    return render(request, "metricas.html", context=metrics_bundle)
