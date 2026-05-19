# ══════════════════════════════════════════════════════════════════════════════
# views.py — SIDM: Sistema Inteligente de Diagnóstico Médico
# Modelo: BETO (dccuchile/bert-base-spanish-wwm-cased)
#
# NOTA: El modelo ya fue entrenado y publicado en Hugging Face.
#       Esta versión NO entrena; descarga los pesos desde HF y los cachea
#       en memoria para todas las peticiones siguientes.
# ══════════════════════════════════════════════════════════════════════════════

# ── Django ────────────────────────────────────────────────────────────────────
from django.shortcuts import render

# ── Librerías base ────────────────────────────────────────────────────────────
from pathlib import Path
import os
import re
import shutil

import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download

# ── TensorFlow ────────────────────────────────────────────────────────────────
import tensorflow as tf

# ── HuggingFace Transformers ──────────────────────────────────────────────────
from transformers import BertConfig, BertTokenizerFast, TFBertForSequenceClassification

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

import pickle
import json

# ══════════════════════════════════════════════════════════════════════════════
# Constantes de configuración
# ══════════════════════════════════════════════════════════════════════════════

BETO_CHECKPOINT = "dccuchile/bert-base-spanish-wwm-cased"
MAX_SEQ_LEN     = 128
HF_REPO_ID      = "CECMHF/BETO_SIDM"

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ── Estado global (singleton) ─────────────────────────────────────────────────
_model: TFBertForSequenceClassification | None = None
_tokenizer_beto: BertTokenizerFast | None = None
_label_encoder: LabelEncoder | None = None


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — Limpieza de texto
# ══════════════════════════════════════════════════════════════════════════════

def normalizar_reporte(report: dict) -> dict:
    """Renombra claves con guion/espacio para que Django templates pueda accederlas."""
    nuevo = {}
    for k, v in report.items():
        nueva_k = k.replace("-", "_").replace(" ", "_")
        if isinstance(v, dict):
            nuevo[nueva_k] = {ik.replace("-", "_"): iv for ik, iv in v.items()}
        else:
            nuevo[nueva_k] = v
    return nuevo


def limpiar_texto(texto: str) -> str:
    """Limpieza mínima para BETO cased: elimina URLs, correos y espacios extra."""
    if pd.isna(texto):
        return ""
    texto = str(texto)
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)
    texto = re.sub(r"\S+@\S+\.\S+", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — Tokenización BETO
# ══════════════════════════════════════════════════════════════════════════════

def encode_texts(
    texts: list,
    tokenizer: BertTokenizerFast,
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte textos en input_ids y attention_mask para BETO."""
    encoding = tokenizer(
        list(texts),
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return encoding["input_ids"], encoding["attention_mask"]


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — Predicción
# ══════════════════════════════════════════════════════════════════════════════

def predecir(
    model: TFBertForSequenceClassification,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (y_pred, y_prob) aplicando softmax sobre los logits del modelo."""
    input_ids      = tf.cast(input_ids,      tf.int32)
    attention_mask = tf.cast(attention_mask, tf.int32)
    token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)

    outputs = model(
        {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        training=False,
    )

    y_prob = tf.nn.softmax(outputs.logits, axis=-1).numpy()
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4 — Carga del modelo desde Hugging Face
# ══════════════════════════════════════════════════════════════════════════════

def get_tokenizer() -> BertTokenizerFast:
    """
    Singleton del tokenizador. Usa el checkpoint original de BETO porque
    el tokenizador no cambia durante el fine-tuning.
    """
    global _tokenizer_beto
    if _tokenizer_beto is None:
        print("Cargando tokenizador desde Hugging Face...")
        _tokenizer_beto = BertTokenizerFast.from_pretrained(BETO_CHECKPOINT)
        print("Tokenizador cargado.")
    return _tokenizer_beto


def cargar_modelo() -> tuple[TFBertForSequenceClassification, BertTokenizerFast, LabelEncoder] | None:
    """
    Carga desde Hugging Face (CECMHF/BETO_SIDM):
      - Modelo         ->  subcarpeta  modelo_beto/
      - Tokenizador    ->  subcarpeta  tokenizer_beto/
      - LabelEncoder   ->  archivo     label_encoder.pkl

    Todo queda cacheado en memoria; solo descarga una vez por proceso.
    Retorna (model, tokenizer, label_encoder) o None si falla.
    """
    global _model, _tokenizer_beto, _label_encoder

    try:
        # ── Modelo ────────────────────────────────────────────────────────────
        if _model is None:
            print("Cargando modelo desde Hugging Face (puede tardar la primera vez)...")

            # El config.json guardado puede tener vocab_size=30522 (BERT inglés)
            # en lugar de 31002 (BETO español), causando un error de reshape al
            # cargar los pesos. Solución: leer num_labels del config guardado y
            # construir la config correcta con el vocab_size real de BETO.
            saved_config = BertConfig.from_pretrained(HF_REPO_ID, subfolder="modelo_beto")
            correct_config = BertConfig.from_pretrained(
                BETO_CHECKPOINT,
                num_labels=saved_config.num_labels,
            )

            _model = TFBertForSequenceClassification.from_pretrained(
                HF_REPO_ID,
                subfolder="modelo_beto",
                config=correct_config,
                from_pt=False,
            )
            print("Modelo cargado correctamente.")

        # ── Tokenizador ───────────────────────────────────────────────────────
        if _tokenizer_beto is None:
            print("Cargando tokenizador desde Hugging Face...")
            _tokenizer_beto = BertTokenizerFast.from_pretrained(
                HF_REPO_ID,
                subfolder="tokenizer_beto",
            )
            print("Tokenizador cargado.")

        # ── Label Encoder ─────────────────────────────────────────────────────
        if _label_encoder is None:
            print("Descargando label_encoder.pkl desde Hugging Face...")
            downloaded_label_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="label_encoder.pkl",
                cache_dir=str(MODEL_DIR),
            )
            with open(downloaded_label_path, "rb") as f:
                _label_encoder = pickle.load(f)
            print("Label encoder cargado.")

        return _model, _tokenizer_beto, _label_encoder

    except Exception as e:
        print(f"ERROR cargando modelo desde HF: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 5 — Vistas Django
# ══════════════════════════════════════════════════════════════════════════════

def main(request):
    """Vista principal — menú de navegación."""
    metrics_existe = os.path.exists(METRICS_PATH)
    modelo_listo   = _model is not None and _label_encoder is not None

    context = {
        "modelo_existe":  modelo_listo,
        "metrics_existe": metrics_existe,
    }
    return render(request, "index.html", context=context)


def entrenar(request):
    """
    Vista 'Modelo' — en producción NO entrena.

    Flujo:
      1. Descarga y cachea el modelo desde Hugging Face.
      2. Si metrics.json no está en disco, lo descarga de HF.
      3. Renderiza las métricas del entrenamiento previo.
    """
    # ── 1. Cargar modelo desde HF ─────────────────────────────────────────
    resultado = cargar_modelo()

    if resultado is None:
        context = {
            "error": "No se pudo cargar el modelo desde Hugging Face. "
                     "Verifica la conexión y que el repositorio sea accesible.",
            "tipo": "entrenar",
        }
        return render(request, "index.html", context=context)

    # ── 2. Obtener metrics.json (disco o HF) ──────────────────────────────
    if not os.path.exists(METRICS_PATH):
        try:
            print("Descargando metrics.json desde Hugging Face...")
            downloaded_metrics_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="metrics.json",
                cache_dir=str(MODEL_DIR),
            )
            shutil.copy(downloaded_metrics_path, METRICS_PATH)
            print("metrics.json descargado.")
        except Exception as e:
            print(f"No se pudo descargar metrics.json: {e}")

    if not os.path.exists(METRICS_PATH):
        context = {
            "error": "Modelo cargado correctamente, pero no se encontraron "
                     "metricas en el repositorio de Hugging Face.",
            "tipo": "entrenar",
        }
        return render(request, "index.html", context=context)

    # ── 3. Leer y renderizar métricas ─────────────────────────────────────
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_bundle = json.load(f)

        if "reporte_clasificacion" in metrics_bundle:
            metrics_bundle["reporte_clasificacion"] = normalizar_reporte(
                metrics_bundle["reporte_clasificacion"]
            )
    except Exception as e:
        context = {"error": f"Error al leer metricas: {e}", "tipo": "entrenar"}
        return render(request, "index.html", context=context)

    context = {
        "metricas":                    metrics_bundle.get("metricas", {}),
        "matriz_confusion":            metrics_bundle.get("matriz_confusion", []),
        "matriz_confusion_etiquetada": metrics_bundle.get("matriz_confusion_etiquetada", []),
        "reporte_clasificacion":       metrics_bundle.get("reporte_clasificacion", {}),
        "prediccion":                  metrics_bundle.get("prediccion", []),
        "clases":                      metrics_bundle.get("clases", []),
        "tipo":                        "entrenar",
        "modelo_desde_hf":             True,
    }

    return render(request, "index.html", context=context)


def diagnosticar(request):
    """
    Vista para hacer diagnósticos usando el modelo cargado desde HF.

    GET  -> muestra el formulario de sintomas.
    POST -> procesa los sintomas y devuelve diagnostico con top-3.
    """
    resultado = cargar_modelo()

    if resultado is None:
        context = {
            "error": "No se pudo cargar el modelo. Verifica la conexion a Hugging Face.",
            "tipo":  "diagnosticar",
        }
        return render(request, "resultado.html", context=context)

    model, tokenizer, label_encoder = resultado

    if request.method == "GET":
        return render(request, "resultado.html", {"tipo": "formulario"})

    # ── POST: procesar sintomas ────────────────────────────────────────────
    sintomas_usuario = request.POST.get("sintomas", "").strip()

    if not sintomas_usuario:
        context = {
            "error": "Por favor ingresa sintomas para diagnosticar.",
            "tipo":  "formulario",
        }
        return render(request, "resultado.html", context=context)

    sintomas_limpios     = limpiar_texto(sintomas_usuario)
    input_ids, attn_mask = encode_texts([sintomas_limpios], tokenizer)
    y_pred, y_prob       = predecir(model, input_ids, attn_mask)

    pred_idx = int(y_pred[0])
    probs    = y_prob[0]

    top3_idx = np.argsort(probs)[::-1][:3]
    top_3 = [
        {
            "nombre":       label_encoder.classes_[idx],
            "probabilidad": float(probs[idx]) * 100,
        }
        for idx in top3_idx[1:]
    ]

    context = {
        "tipo":                   "diagnostico",
        "sintomas_usuario":       sintomas_usuario,
        "enfermedad_principal":   label_encoder.classes_[pred_idx],
        "probabilidad_principal": float(probs[pred_idx]) * 100,
        "top_3":                  top_3,
    }

    return render(request, "resultado.html", context=context)


def mostrar_metricas(request):
    """Vista para mostrar las metricas del entrenamiento guardadas en disco."""
    if not os.path.exists(METRICS_PATH):
        context = {"error": "No hay metricas guardadas. Visita /entrenar/ primero."}
        return render(request, "metricas.html", context=context)

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_bundle = json.load(f)

        if "reporte_clasificacion" in metrics_bundle:
            metrics_bundle["reporte_clasificacion"] = normalizar_reporte(
                metrics_bundle["reporte_clasificacion"]
            )

    except Exception as e:
        context = {"error": f"Error al leer metricas: {e}"}
        return render(request, "metricas.html", context=context)

    return render(request, "metricas.html", context=metrics_bundle)