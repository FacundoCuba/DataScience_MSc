#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorización Masiva de Secuencias Proteicas mediante ESM-2 (pLLM) en CUDA.

Extrae las representaciones latentes (embeddings continuos) para el conjunto completo 
de secuencias en 'genes_curados' (690,579) utilizando el lenguaje de modelado proteico 
ESM-2 (Protein Large Language Model) optimizado mediante cómputo distribuido en GPU 
(PyTorch / HuggingFace Transformers).

Flujo de trabajo:
-----------------
1. Carga del modelo de lenguaje proteico ESM-2 (`facebook/esm2_t12_35M_UR50D`) y su tokenizador.
2. Recuperación en streaming mediante cursores `no_cursor_timeout` de MongoDB desde la 
   colección origen (`genes_curados`).
3. Tokenización dinámica por lotes con truncamiento explícito en 1,175 aminoácidos y 
   generación automática de máscaras de atención (`attention_mask`).
4. Inferencia del estado oculto (*last hidden state*) acelerada por GPU usando Precisión 
   Mixta Automática (FP16 via `torch.amp.autocast`).
5. Agregación espacio-secuencial mediante *Mean Pooling* en GPU, ponderando únicamente los 
   residuos válidos (descartando tokens de padding).
6. Persistencia masiva de vectores (`esm2_vector`) e indexación única por `protein_id` en 
   la colección destino (`vec_esm2`).
"""

import gc
import time
import torch
from pymongo import MongoClient
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

def run_esm2_vectorization_full(
    model_name: str = "facebook/esm2_t12_35M_UR50D", 
    batch_size: int = 64,
    max_length: int = 1175
) -> None:
    """Ejecuta la canalización de vectorización masiva con ESM-2 para 690,579 genes.

    Gestiona la conexión streaming con MongoDB, la carga e inferencia del modelo pLLM
    en batches optimizados para GPU y la creación posterior de índices.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] Dispositivo asignado: {device}")

    # Cargar Tokenizer y Modelo ESM-2
    print(f"[{time.strftime('%H:%M:%S')}] Cargando pLLM '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]
    src_col = db["genes_curados"]
    dst_col = db["vec_esm2"]

    total_docs = src_col.count_documents({})
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando vectorización masiva pLLM. Total genes: {total_docs:,}")
    dst_col.drop()

    # Cursor streaming robusto sin límite de tiempo
    cursor = src_col.find(
        {}, 
        {"protein_id": 1, "aa_sequence": 1, "_id": 0}, 
        no_cursor_timeout=True
    ).batch_size(batch_size * 2)

    inserted = 0
    batch_docs = []

    try:
        with tqdm(total=total_docs, desc="ESM-2 Embeddings (FP16)", unit="seq") as pbar:
            for doc in cursor:
                # Filtrar secuencias vacías o inválidas
                if doc.get("protein_id") and doc.get("aa_sequence"):
                    batch_docs.append(doc)

                if len(batch_docs) >= batch_size:
                    _process_and_insert_esm_batch(
                        batch_docs, model, tokenizer, device, dst_col, model_name, max_length
                    )
                    inserted += len(batch_docs)
                    pbar.update(len(batch_docs))
                    batch_docs = []

            # Procesar residuos finales
            if batch_docs:
                _process_and_insert_esm_batch(
                    batch_docs, model, tokenizer, device, dst_col, model_name, max_length
                )
                inserted += len(batch_docs)
                pbar.update(len(batch_docs))

    finally:
        cursor.close()

    print(f"[{time.strftime('%H:%M:%S')}] Indexando colección 'vec_esm2'...")
    dst_col.create_index("protein_id")
    print(f"[{time.strftime('%H:%M:%S')}] Vectorización ESM-2 finalizada exitosamente. Insertados: {inserted:,} documentos.")

def _process_and_insert_esm_batch(
    batch_docs: list[dict],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    dst_col,
    model_name: str,
    max_length: int
) -> None:
    """Procesa un lote de secuencias con ESM-2 en FP16, aplica Mean Pooling y persiste en MongoDB.

    Tokeniza las secuencias, ejecuta la pasada hacia adelante bajo precisión mixta 
    (`torch.amp.autocast`), reduce los estados ocultos promediando sobre la dimensión 
    secuencial (descartando el relleno) e inserta los embeddings en la base de datos.
    """
    sequences = [d["aa_sequence"] for d in batch_docs]
    protein_ids = [d["protein_id"] for d in batch_docs]

    # Tokenización con truncado explícito y padding dinámico
    inputs = tokenizer(
        sequences, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    ).to(device)

    # Inferencia optimizada con Precision Mixta (FP16)
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            # Mean pooling ignorando tokens de padding
            sum_embeddings = torch.sum(last_hidden_states * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).to(torch.float32).cpu().numpy()

    # Armado del payload para MongoDB
    mongo_batch = [
        {
            "protein_id": p_id,
            "esm2_vector": vec.tolist(),
            "model_name": model_name
        }
        for p_id, vec in zip(protein_ids, mean_pooled)
    ]
    
    dst_col.insert_many(mongo_batch, ordered=False)

if __name__ == "__main__":
    # Batch size 64 funciona impecablemente en GPUs de 12GB como la RTX 5070 con FP16
    run_esm2_vectorization_full(
        model_name="facebook/esm2_t12_35M_UR50D", 
        batch_size=64, 
        max_length=1175
    )