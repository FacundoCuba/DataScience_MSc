#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorización Masiva de Secuencias Proteicas mediante ProtTrans (ProtT5-XL-Half) en CUDA.

Optimizado para GPUs de 12GB VRAM en WSL2 mediante Token Budgeting, 
reanudación automática desde MongoDB y torch.inference_mode().
Utiliza la versión 'half-precision' del encoder ProtT5 para un consumo mínimo de VRAM.
"""

import os
import gc
import re
import time
import torch
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm

# Configuración de asignación de memoria compatible con WSL2 (evita colapsos de handles en CUDA)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

try:
    from transformers import T5Tokenizer, T5EncoderModel
except ImportError:
    raise ImportError("No se encontró el paquete 'transformers'. Instálalo con: pip install transformers sentencepiece")


def run_prott5_vectorization_full(
    model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc", 
    max_tokens_per_batch: int = 1200,
    max_length: int = 1000
) -> None:
    """Ejecuta la vectorización masiva reanudando el progreso si ya existen vectores guardados en MongoDB.

    Parameters
    ----------
    model_name : str
        Identificador del modelo en Hugging Face (por defecto 'Rostlab/prot_t5_xl_half_uniref50-enc').
    max_tokens_per_batch : int
        Presupuesto de tokens por lote para evitar Out-Of-Memory (OOM) en VRAM.
    max_length : int
        Longitud máxima de aminoácidos por secuencia a procesar.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] Dispositivo asignado: {device}")

    print(f"[{time.strftime('%H:%M:%S')}] Cargando pLLM ProtT5 ('{model_name}')...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()

    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]
    src_col = db["genes_curados"]
    dst_col = db["vec_prott5"]

    # Asegurar índice para acelerar consultas y evitar duplicados
    dst_col.create_index("protein_id")

    # Reanudación: identificar IDs ya vectorizados previamente
    existing_ids = set(dst_col.distinct("protein_id"))
    if existing_ids:
        print(f"[{time.strftime('%H:%M:%S')}] Reanudando ejecución. Registros previos en MongoDB: {len(existing_ids):,}")

    query = {"protein_id": {"$nin": list(existing_ids)}} if existing_ids else {}
    total_docs = src_col.count_documents(query)
    print(f"[{time.strftime('%H:%M:%S')}] Secuencias pendientes de vectorizar: {total_docs:,}")

    if total_docs == 0:
        print(f"[{time.strftime('%H:%M:%S')}] No hay nuevas secuencias para procesar.")
        return

    cursor = src_col.find(
        query, 
        {"protein_id": 1, "aa_sequence": 1, "_id": 0}, 
        no_cursor_timeout=True
    ).batch_size(100)

    inserted = 0
    current_batch = []
    current_tokens = 0

    try:
        with tqdm(total=total_docs, desc="ProtT5 Embeddings", unit="seq") as pbar:
            for doc in cursor:
                if doc.get("protein_id") and doc.get("aa_sequence"):
                    seq = doc["aa_sequence"][:max_length]
                    seq_len = len(seq)

                    # Procesar lote si excede el presupuesto de tokens
                    if current_batch and (current_tokens + seq_len > max_tokens_per_batch):
                        _safe_process_and_insert(current_batch, model, tokenizer, device, dst_col, model_name)
                        inserted += len(current_batch)
                        pbar.update(len(current_batch))
                        current_batch = []
                        current_tokens = 0

                    current_batch.append({"protein_id": doc["protein_id"], "aa_sequence": seq})
                    current_tokens += seq_len

            # Procesar el lote final
            if current_batch:
                _safe_process_and_insert(current_batch, model, tokenizer, device, dst_col, model_name)
                inserted += len(current_batch)
                pbar.update(len(current_batch))

    finally:
        cursor.close()

    print(f"[{time.strftime('%H:%M:%S')}] Vectorización completada. Nuevos documentos insertados: {inserted:,}")


def _safe_process_and_insert(
    batch_docs: list[dict],
    model: torch.nn.Module,
    tokenizer,
    device: str,
    dst_col,
    model_name: str
) -> None:
    """Procesa un lote; si ocurre un error de VRAM o PyTorch, recurre al procesamiento individual."""
    try:
        _process_and_insert_prott5_batch(batch_docs, model, tokenizer, device, dst_col, model_name)
    except Exception as e:
        if "CUDA" in str(e) or "out of memory" in str(e).lower() or "INTERNAL ASSERT FAILED" in str(e):
            if device == "cuda":
                torch.cuda.empty_cache()
            # Fallback a 1 por 1
            for single_doc in batch_docs:
                _process_and_insert_prott5_batch([single_doc], model, tokenizer, device, dst_col, model_name)
        else:
            raise e


def _process_and_insert_prott5_batch(
    batch_docs: list[dict],
    model: torch.nn.Module,
    tokenizer,
    device: str,
    dst_col,
    model_name: str
) -> None:
    """Inferencia optimizada en PyTorch con separación de residuos y mean-pooling."""
    protein_ids = [d["protein_id"] for d in batch_docs]
    
    # ProtT5 requiere que los aminoácidos estén separados por espacios
    # y mapear aminoácidos no estándar (U, Z, O, B) a 'X'
    clean_sequences = [
        " ".join(list(re.sub(r"[UZOB]", "X", d["aa_sequence"].upper()))) 
        for d in batch_docs
    ]

    inputs = tokenizer(
        clean_sequences, 
        add_special_tokens=True, 
        padding=True, 
        return_tensors="pt"
    ).to(device)

    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=amp_dtype):
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            token_embeddings = outputs.last_hidden_state

            # Mean pooling ignorando tokens de padding y tokens especiales
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).to(torch.float32).cpu().numpy()

    mongo_batch = [
        {
            "protein_id": p_id,
            "prott5_vector": vec.tolist(),
            "model_name": model_name
        }
        for p_id, vec in zip(protein_ids, mean_pooled)
    ]

    dst_col.insert_many(mongo_batch, ordered=False)

    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_prott5_vectorization_full(
        model_name="Rostlab/prot_t5_xl_half_uniref50-enc", 
        max_tokens_per_batch=1200, 
        max_length=1000
    )