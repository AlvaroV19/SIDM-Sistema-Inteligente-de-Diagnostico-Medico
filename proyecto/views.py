# ══════════════════════════════════════════════════════════════════════════════
# views.py — SIDM: Sistema Inteligente de Diagnóstico Médico
# ══════════════════════════════════════════════════════════════════════════════

from django.shortcuts import render

from pathlib import Path
import os
import re
import shutil
import json
import pickle

import numpy as np

from huggingface_hub import hf_hub_download


# ══════════════════════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════════════════════

BETO_CHECKPOINT = "dccuchile/bert-base-spanish-wwm-cased"
MAX_SEQ_LEN = 128
HF_REPO_ID = "CECMHF/BETO_SIDM"

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


# ══════════════════════════════════════════════════════════════════════════════
# Estado global cacheado
# ══════════════════════════════════════════════════════════════════════════════

_model = None
_tokenizer_beto = None
_label_encoder = None


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades
# ══════════════════════════════════════════════════════════════════════════════

def normalizar_reporte(report: dict) -> dict:
    nuevo = {}

    for k, v in report.items():
        nueva_k = k.replace("-", "_").replace(" ", "_")

        if isinstance(v, dict):
            nuevo[nueva_k] = {
                ik.replace("-", "_"): iv
                for ik, iv in v.items()
            }
        else:
            nuevo[nueva_k] = v

    return nuevo


def limpiar_texto(texto: str) -> str:
    if texto is None:
        return ""

    texto = str(texto)

    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)
    texto = re.sub(r"\S+@\S+\.\S+", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


# ══════════════════════════════════════════════════════════════════════════════
# Tokenización
# ══════════════════════════════════════════════════════════════════════════════

def encode_texts(texts, tokenizer):

    encoding = tokenizer(
        list(texts),
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    return encoding["input_ids"], encoding["attention_mask"]


# ══════════════════════════════════════════════════════════════════════════════
# Predicción
# ══════════════════════════════════════════════════════════════════════════════

def predecir(model, input_ids, attention_mask):

    import tensorflow as tf

    input_ids = tf.cast(input_ids, tf.int32)
    attention_mask = tf.cast(attention_mask, tf.int32)

    token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)

    outputs = model(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        training=False,
    )

    y_prob = tf.nn.softmax(outputs.logits, axis=-1).numpy()
    y_pred = np.argmax(y_prob, axis=1)

    return y_pred, y_prob


# ══════════════════════════════════════════════════════════════════════════════
# Carga del modelo
# ══════════════════════════════════════════════════════════════════════════════

def cargar_modelo():

    global _model
    global _tokenizer_beto
    global _label_encoder

    try:

        # Lazy imports → reducen RAM al iniciar Gunicorn
        from transformers import (
            BertConfig,
            BertTokenizerFast,
            TFBertForSequenceClassification,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Modelo
        # ─────────────────────────────────────────────────────────────────────

        if _model is None:

            print("Cargando modelo desde Hugging Face...")

            saved_config = BertConfig.from_pretrained(
                HF_REPO_ID,
                subfolder="modelo_beto",
            )

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

        # ─────────────────────────────────────────────────────────────────────
        # Tokenizador
        # ─────────────────────────────────────────────────────────────────────

        if _tokenizer_beto is None:

            print("Cargando tokenizador...")

            _tokenizer_beto = BertTokenizerFast.from_pretrained(
                HF_REPO_ID,
                subfolder="tokenizer_beto",
            )

            print("Tokenizador cargado.")

        # ─────────────────────────────────────────────────────────────────────
        # Label encoder
        # ─────────────────────────────────────────────────────────────────────

        if _label_encoder is None:

            print("Descargando label_encoder.pkl...")

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

        print(f"ERROR cargando modelo: {e}")

        return None


# ══════════════════════════════════════════════════════════════════════════════
# Views Django
# ══════════════════════════════════════════════════════════════════════════════

def main(request):

    metrics_existe = os.path.exists(METRICS_PATH)

    modelo_listo = (
        _model is not None
        and _label_encoder is not None
    )

    context = {
        "modelo_existe": modelo_listo,
        "metrics_existe": metrics_existe,
    }

    return render(request, "index.html", context=context)


def entrenar(request):

    resultado = cargar_modelo()

    if resultado is None:

        context = {
            "error": (
                "No se pudo cargar el modelo desde Hugging Face."
            ),
            "tipo": "entrenar",
        }

        return render(request, "index.html", context=context)

    # ─────────────────────────────────────────────────────────────────────────
    # Descargar métricas
    # ─────────────────────────────────────────────────────────────────────────

    if not os.path.exists(METRICS_PATH):

        try:

            print("Descargando metrics.json...")

            downloaded_metrics_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="metrics.json",
                cache_dir=str(MODEL_DIR),
            )

            shutil.copy(
                downloaded_metrics_path,
                METRICS_PATH,
            )

        except Exception as e:

            print(f"Error descargando métricas: {e}")

    if not os.path.exists(METRICS_PATH):

        context = {
            "error": "No se encontraron métricas.",
            "tipo": "entrenar",
        }

        return render(request, "index.html", context=context)

    # ─────────────────────────────────────────────────────────────────────────
    # Leer métricas
    # ─────────────────────────────────────────────────────────────────────────

    try:

        with open(METRICS_PATH, "r", encoding="utf-8") as f:

            metrics_bundle = json.load(f)

        if "reporte_clasificacion" in metrics_bundle:

            metrics_bundle["reporte_clasificacion"] = (
                normalizar_reporte(
                    metrics_bundle["reporte_clasificacion"]
                )
            )

    except Exception as e:

        context = {
            "error": f"Error leyendo métricas: {e}",
            "tipo": "entrenar",
        }

        return render(request, "index.html", context=context)

    context = {
        "metricas": metrics_bundle.get("metricas", {}),
        "matriz_confusion": metrics_bundle.get("matriz_confusion", []),
        "matriz_confusion_etiquetada": metrics_bundle.get(
            "matriz_confusion_etiquetada", []
        ),
        "reporte_clasificacion": metrics_bundle.get(
            "reporte_clasificacion", {}
        ),
        "prediccion": metrics_bundle.get("prediccion", []),
        "clases": metrics_bundle.get("clases", []),
        "tipo": "entrenar",
        "modelo_desde_hf": True,
    }

    return render(request, "index.html", context=context)


def diagnosticar(request):

    resultado = cargar_modelo()

    if resultado is None:

        context = {
            "error": (
                "No se pudo cargar el modelo."
            ),
            "tipo": "diagnosticar",
        }

        return render(request, "resultado.html", context=context)

    model, tokenizer, label_encoder = resultado

    if request.method == "GET":

        return render(
            request,
            "resultado.html",
            {"tipo": "formulario"},
        )

    sintomas_usuario = request.POST.get(
        "sintomas",
        "",
    ).strip()

    if not sintomas_usuario:

        context = {
            "error": "Por favor ingresa síntomas.",
            "tipo": "formulario",
        }

        return render(
            request,
            "resultado.html",
            context=context,
        )

    sintomas_limpios = limpiar_texto(
        sintomas_usuario
    )

    input_ids, attn_mask = encode_texts(
        [sintomas_limpios],
        tokenizer,
    )

    y_pred, y_prob = predecir(
        model,
        input_ids,
        attn_mask,
    )

    pred_idx = int(y_pred[0])

    probs = y_prob[0]

    # Top 1 principal + 3 alternativas
    top4_idx = np.argsort(probs)[::-1][:4]

    top_3 = [
        {
            "nombre": label_encoder.classes_[idx],
            "probabilidad": float(probs[idx]) * 100,
        }
        for idx in top4_idx[1:]
    ]

    context = {
        "tipo": "diagnostico",
        "sintomas_usuario": sintomas_usuario,
        "enfermedad_principal": (
            label_encoder.classes_[pred_idx]
        ),
        "probabilidad_principal": (
            float(probs[pred_idx]) * 100
        ),
        "top_3": top_3,
    }

    return render(
        request,
        "resultado.html",
        context=context,
    )


def mostrar_metricas(request):

    if not os.path.exists(METRICS_PATH):

        context = {
            "error": (
                "No hay métricas guardadas."
            )
        }

        return render(
            request,
            "metricas.html",
            context=context,
        )

    try:

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            metrics_bundle = json.load(f)

        if "reporte_clasificacion" in metrics_bundle:

            metrics_bundle["reporte_clasificacion"] = (
                normalizar_reporte(
                    metrics_bundle["reporte_clasificacion"]
                )
            )

    except Exception as e:

        context = {
            "error": f"Error leyendo métricas: {e}"
        }

        return render(
            request,
            "metricas.html",
            context=context,
        )

    return render(
        request,
        "metricas.html",
        context=metrics_bundle,
    )
