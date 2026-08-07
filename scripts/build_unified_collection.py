#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Consolidación de Representaciones Vectoriales en MongoDB (Feature Store).

Este script unifica en una sola colección ('genes_unified_features') los metadatos
de anotación funcional ('product') de 'genes_curados' junto con sus tres espacios
vectoriales correspondientes ('vec_kmers', 'vec_align_mds' y 'vec_esm2').

El proceso realiza un join eficiente en memoria por lotes (chunks) utilizando
'protein_id' como clave foránea, garantizando la consistencia 1:1 de los vectores
y optimizando los I/O para experimentos de fusión.
"""

import time
from pymongo import MongoClient
from tqdm import tqdm

def build_unified_collection(chunk_size: int = 5000):
    """Ejecuta el pipeline de consolidación masiva de características proteicas.

    Limpia la colección de destino 'genes_unified_features', extrae en streaming
    los registros con anotación funcional ('product') y 'protein_id' válidos desde
    'genes_curados', procesa la integración vectorial por lotes y genera índices
    secundarios para acelerar consultas de benchmark.
    """
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]
    
    src_curados = db["genes_curados"]
    col_kmers = db["vec_kmers"]
    col_mds = db["vec_align_mds"]
    col_esm2 = db["vec_esm2"]
    dst_col = db["genes_unified_features"]

    print(f"[{time.strftime('%H:%M:%S')}] Reiniciando colección 'genes_unified_features'...")
    dst_col.drop()

    # Filtrar solo genes que tengan anotación funcional 'product'
    query = {"product": {"$ne": None}, "protein_id": {"$ne": None}}
    total_docs = src_curados.count_documents(query)
    print(f"[{time.strftime('%H:%M:%S')}] Consolidando {total_docs:,} genes con anotación...")

    cursor = src_curados.find(query, {"protein_id": 1, "product": 1, "_id": 0}, no_cursor_timeout=True).batch_size(chunk_size)
    
    inserted = 0
    batch_docs = []

    try:
        with tqdm(total=total_docs, desc="Consolidando BBDD", unit="seq") as pbar:
            for doc in cursor:
                batch_docs.append(doc)
                if len(batch_docs) >= chunk_size:
                    _process_and_insert_unified_batch(batch_docs, col_kmers, col_mds, col_esm2, dst_col)
                    inserted += len(batch_docs)
                    pbar.update(len(batch_docs))
                    batch_docs = []

            if batch_docs:
                _process_and_insert_unified_batch(batch_docs, col_kmers, col_mds, col_esm2, dst_col)
                inserted += len(batch_docs)
                pbar.update(len(batch_docs))
    finally:
        cursor.close()

    print(f"[{time.strftime('%H:%M:%S')}] Indexando 'protein_id' y 'product'...")
    dst_col.create_index("protein_id")
    dst_col.create_index("product")
    print(f"[{time.strftime('%H:%M:%S')}] Consolidación completada. Total: {inserted:,} documentos.")

def _process_and_insert_unified_batch(batch_docs, col_kmers, col_mds, col_esm2, dst_col):
    """Recupera, integra y persiste un lote de documentos consolidados en MongoDB.

    Obtiene en memoria los tres vectores asociados a los 'protein_id' del lote
    desde sus respectivas colecciones ('vec_kmers', 'vec_align_mds', 'vec_esm2').
    Filtra las proteínas que cuentan con representación completa en los tres espacios
    y realiza una inserción masiva en la colección destino.
    """
    p_ids = [d["protein_id"] for d in batch_docs]

    # Indexar vectores del lote en memoria
    kmers_map = {d["protein_id"]: d["kmer_vector"] for d in col_kmers.find({"protein_id": {"$in": p_ids}}, {"protein_id": 1, "kmer_vector": 1, "_id": 0})}
    mds_map = {d["protein_id"]: d["align_mds_vector"] for d in col_mds.find({"protein_id": {"$in": p_ids}}, {"protein_id": 1, "align_mds_vector": 1, "_id": 0})}
    esm2_map = {d["protein_id"]: d["esm2_vector"] for d in col_esm2.find({"protein_id": {"$in": p_ids}}, {"protein_id": 1, "esm2_vector": 1, "_id": 0})}

    unified_batch = []
    for d in batch_docs:
        p_id = d["protein_id"]
        # Se requiere que la proteína exista en las tres colecciones vectoriales
        if p_id in kmers_map and p_id in mds_map and p_id in esm2_map:
            unified_batch.append({
                "protein_id": p_id,
                "product": d["product"],
                "kmer_vector": kmers_map[p_id],
                "align_mds_vector": mds_map[p_id],
                "esm2_vector": esm2_map[p_id]
            })

    if unified_batch:
        dst_col.insert_many(unified_batch, ordered=False)

if __name__ == "__main__":
    build_unified_collection()