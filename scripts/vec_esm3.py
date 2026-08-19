#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorización Masiva de Secuencias Proteicas mediante ESM-3 (pLLM) en CUDA.

Optimizado para GPUs de 12GB VRAM en WSL2 mediante Token Budgeting, 
reanudación automática desde MongoDB y torch.inference_mode().
"""

import os
import gc
import time
import torch
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm

# Configuración de asignación de memoria compatible con WSL2 (evita colapsos de handles en CUDA)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

try:
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
except ImportError:
    raise ImportError("No se encontró el paquete 'esm'. Instálalo con: pip install esm")


def run_esm3_vectorization_full(
    model_name: str = "esm3_sm_open_v1", 
    max_tokens_per_batch: int = 500,
    max_length: int = 1000
) -> None:
    """Ejecuta la vectorización masiva reanudando el progreso si ya existen vectores guardados."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] Dispositivo asignado: {device}")

    print(f"[{time.strftime('%H:%M:%S')}] Cargando pLLM ESM-3 ('{model_name}')...")
    model = ESM3.from_pretrained(model_name).to(device)
    model.eval()

    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]
    src_col = db["genes_curados"]
    dst_col = db["vec_esm3"]

    # Asegurar índice para acelerar consultas y evitar duplicados
    dst_col.create_index("protein_id")

    # Reanudación: extraer IDs evitando el límite BSON de 16MB de .distinct()
    print(f"[{time.strftime('%H:%M:%S')}] Escaneando IDs ya procesados en MongoDB...")
    existing_ids = {
        doc["protein_id"] 
        for doc in dst_col.find({}, {"protein_id": 1, "_id": 0}) 
        if "protein_id" in doc
    }
    if existing_ids:
        print(f"[{time.strftime('%H:%M:%S')}] Reanudando ejecución. Registros previas en MongoDB: {len(existing_ids):,}")

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
        with tqdm(total=total_docs, desc="ESM-3 Embeddings", unit="seq") as pbar:
            for doc in cursor:
                if doc.get("protein_id") and doc.get("aa_sequence"):
                    seq = doc["aa_sequence"][:max_length]
                    seq_len = len(seq)

                    # Procesar lote si excede el presupuesto de tokens
                    if current_batch and (current_tokens + seq_len > max_tokens_per_batch):
                        _safe_process_and_insert(current_batch, model, device, dst_col, model_name)
                        inserted += len(current_batch)
                        pbar.update(len(current_batch))
                        current_batch = []
                        current_tokens = 0

                    current_batch.append({"protein_id": doc["protein_id"], "aa_sequence": seq})
                    current_tokens += seq_len

            # Procesar el lote final
            if current_batch:
                _safe_process_and_insert(current_batch, model, device, dst_col, model_name)
                inserted += len(current_batch)
                pbar.update(len(current_batch))

    finally:
        cursor.close()

    print(f"[{time.strftime('%H:%M:%S')}] Vectorización completada. Nuevos documentos insertados: {inserted:,}")


def _safe_process_and_insert(
    batch_docs: list[dict],
    model: torch.nn.Module,
    device: str,
    dst_col,
    model_name: str
) -> None:
    """Procesa un lote; si ocurre un error de VRAM o PyTorch, recurre al procesamiento individual."""
    try:
        _process_and_insert_esm3_batch(batch_docs, model, device, dst_col, model_name)
    except Exception as e:
        if "CUDA" in str(e) or "out of memory" in str(e).lower() or "INTERNAL ASSERT FAILED" in str(e):
            if device == "cuda":
                torch.cuda.empty_cache()
            # Fallback a 1 por 1
            for single_doc in batch_docs:
                _process_and_insert_esm3_batch([single_doc], model, device, dst_col, model_name)
        else:
            raise e


def _process_and_insert_esm3_batch(
    batch_docs: list[dict],
    model: torch.nn.Module,
    device: str,
    dst_col,
    model_name: str
) -> None:
    """Inferencia optimizada en PyTorch con autocast y padding manual."""
    sequences = [d["aa_sequence"] for d in batch_docs]
    protein_ids = [d["protein_id"] for d in batch_docs]

    proteins = [ESMProtein(sequence=seq) for seq in sequences]
    tokenized_list = [model.encode(p) for p in proteins]

    max_batch_len = max(t.sequence.shape[0] for t in tokenized_list)
    
    seq_tokenizer = model.tokenizers.sequence
    pad_idx = getattr(seq_tokenizer, "pad_token_id", getattr(seq_tokenizer, "pad_idx", 0))

    batch_tokens = []
    masks = []

    for t in tokenized_list:
        tokens = t.sequence
        seq_len = tokens.shape[0]
        
        padded = torch.full((max_batch_len,), pad_idx, dtype=torch.long)
        padded[:seq_len] = tokens
        
        mask = torch.zeros(max_batch_len, dtype=torch.float32)
        if seq_len > 2:
            mask[1:seq_len-1] = 1.0
        else:
            mask[:seq_len] = 1.0

        batch_tokens.append(padded)
        masks.append(mask)

    input_tokens = torch.stack(batch_tokens).to(device)
    attention_mask = torch.stack(masks).to(device).unsqueeze(-1)

    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    
    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=amp_dtype):
            output = model(sequence_tokens=input_tokens)
            
            if hasattr(output, "embeddings"):
                hidden_states = output.embeddings
            elif isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = getattr(output, "last_hidden_state", output)

            sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).to(torch.float32).cpu().numpy()

    mongo_batch = [
        {
            "protein_id": p_id,
            "esm3_vector": vec.tolist(),
            "model_name": model_name
        }
        for p_id, vec in zip(protein_ids, mean_pooled)
    ]

    dst_col.insert_many(mongo_batch, ordered=False)

    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_esm3_vectorization_full(
        model_name="esm3_sm_open_v1", 
        max_tokens_per_batch=500, 
        max_length=1000
    )