#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Consolidación de Representaciones Vectoriales en MongoDB (Feature Store).

Unifica en una sola colección ('genes_unified_features') los metadatos
de anotación funcional ('product') de 'genes_curados' junto con sus cuatro espacios
vectoriales correspondientes ('vec_kmers6', 'vec_align_mds', 'vec_esm3' y 'vec_prott5').

Estrategia:
- Extrae la clave del vector dinámicamente buscando nombres conocidos (p. ej., 'kmer_vector', 'vector', etc.).
- Filtra e inserta solo aquellas proteínas que cuenten con representación completa en los 4 espacios (intersección 1:1).
"""

import time
from pymongo import MongoClient
from tqdm import tqdm


def _extract_vector_field(doc: dict, candidates: list[str]):
    """Busca y extrae la clave del vector dentro de un documento según una lista de candidatos."""
    if not doc:
        return None
    for cand in candidates:
        if cand in doc and doc[cand] is not None:
            return doc[cand]
    return None


def build_unified_collection(chunk_size: int = 5000):
    """Ejecuta el pipeline de consolidación masiva de características proteicas."""
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    src_curados = db["genes_curados"]
    col_kmers6 = db["vec_kmers6"]
    col_mds = db["vec_align_mds"]
    col_esm3 = db["vec_esm3"]
    col_prott5 = db["vec_prott5"]

    dst_col = db["genes_unified_features"]

    print(f"[{time.strftime('%H:%M:%S')}] Reiniciando colección 'genes_unified_features'...")
    dst_col.drop()

    # Filtrar solo genes que tengan anotación funcional 'product' y 'protein_id' válidos
    query = {"product": {"$ne": None, "$ne": ""}, "protein_id": {"$ne": None, "$ne": ""}}
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
                    inserted += _process_and_insert_unified_batch(
                        batch_docs, col_kmers6, col_mds, col_esm3, col_prott5, dst_col
                    )
                    pbar.update(len(batch_docs))
                    batch_docs = []

            if batch_docs:
                inserted += _process_and_insert_unified_batch(
                    batch_docs, col_kmers6, col_mds, col_esm3, col_prott5, dst_col
                )
                pbar.update(len(batch_docs))
    finally:
        cursor.close()

    print(f"[{time.strftime('%H:%M:%S')}] Creando índices en 'protein_id' y 'product'...")
    dst_col.create_index("protein_id")
    dst_col.create_index("product")
    print(f"[{time.strftime('%H:%M:%S')}] Consolidación completada. Total insertados con 4 vectores: {inserted:,} documentos.")


def _process_and_insert_unified_batch(batch_docs, col_kmers6, col_mds, col_esm3, col_prott5, dst_col) -> int:
    """Recupera, integra y persiste un lote de documentos consolidados en MongoDB."""
    p_ids = [d["protein_id"] for d in batch_docs]

    # Candidatos de nombres de campos posibles en cada colección
    kmers_cands = ["kmer_vector", "kmer6_vector", "kmers6_vector", "vector"]
    mds_cands = ["align_mds_vector", "vector"]
    esm3_cands = ["esm3_vector", "vector"]
    prott5_cands = ["prott5_vector", "vector"]

    # Consultar lotes en MongoDB
    raw_kmers = list(col_kmers6.find({"protein_id": {"$in": p_ids}}, {"_id": 0}))
    raw_mds = list(col_mds.find({"protein_id": {"$in": p_ids}}, {"_id": 0}))
    raw_esm3 = list(col_esm3.find({"protein_id": {"$in": p_ids}}, {"_id": 0}))
    raw_prott5 = list(col_prott5.find({"protein_id": {"$in": p_ids}}, {"_id": 0}))

    # Mapear por protein_id extrayendo el vector dinámicamente
    kmers_map = {d["protein_id"]: _extract_vector_field(d, kmers_cands) for d in raw_kmers if "protein_id" in d}
    mds_map = {d["protein_id"]: _extract_vector_field(d, mds_cands) for d in raw_mds if "protein_id" in d}
    esm3_map = {d["protein_id"]: _extract_vector_field(d, esm3_cands) for d in raw_esm3 if "protein_id" in d}
    prott5_map = {d["protein_id"]: _extract_vector_field(d, prott5_cands) for d in raw_prott5 if "protein_id" in d}

    unified_batch = []
    for d in batch_docs:
        p_id = d["protein_id"]
        vec_kmers = kmers_map.get(p_id)
        vec_mds = mds_map.get(p_id)
        vec_esm3 = esm3_map.get(p_id)
        vec_prott5 = prott5_map.get(p_id)

        # Requisito estricto: la proteína debe tener vector válido en las 4 colecciones
        if vec_kmers is not None and vec_mds is not None and vec_esm3 is not None and vec_prott5 is not None:
            unified_batch.append(
                {
                    "protein_id": p_id,
                    "product": d["product"],
                    "kmers6_vector": vec_kmers,
                    "align_mds_vector": vec_mds,
                    "esm3_vector": vec_esm3,
                    "prott5_vector": vec_prott5,
                }
            )

    if unified_batch:
        dst_col.insert_many(unified_batch, ordered=False)
        return len(unified_batch)

    return 0


if __name__ == "__main__":
    build_unified_collection()