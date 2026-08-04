#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorización masiva de secuencias proteicas mediante estadística de composición (K-mers).

Procesa el total de secuencias en 'genes_curados' (690,579) y almacena 
los vectores de frecuencias relativas en la colección 'vec_kmers'.
"""

from collections import Counter
from pymongo import MongoClient
from tqdm import tqdm
import time


def extract_kmers(sequence: str, k: int = 3) -> dict[str, float]:
    """Calcula la frecuencia relativa de k-mers en una secuencia de aminoácidos."""
    n = len(sequence)
    if not sequence or n < k:
        return {}
    
    total_kmers = n - k + 1
    kmers_counts = Counter(sequence[i:i+k] for i in range(total_kmers))
    return {kmer: count / total_kmers for kmer, count in kmers_counts.items()}


def run_kmers_vectorization_full(k: int = 3, chunk_size: int = 10000) -> None:
    """Procesa masivamente los 690,579 genes y los persiste en 'vec_kmers'."""
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]
    src_col = db["genes_curados"]
    dst_col = db["vec_kmers"]

    total_docs = src_col.count_documents({})
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando vectorización masiva por {k}-mers. Total genes: {total_docs:,}")
    
    # Reiniciamos la colección destino
    dst_col.drop()

    # Cursor robusto para evitar desconexiones en operaciones largas
    cursor = src_col.find(
        {}, 
        {"protein_id": 1, "aa_sequence": 1, "_id": 0},
        no_cursor_timeout=True
    ).batch_size(chunk_size)

    batch = []
    inserted = 0

    try:
        with tqdm(total=total_docs, desc=f"K-mers (k={k})", unit="seq") as pbar:
            for doc in cursor:
                p_id = doc.get("protein_id")
                seq = doc.get("aa_sequence", "")
                
                if p_id and seq:
                    kmer_vec = extract_kmers(seq, k=k)
                    batch.append({
                        "protein_id": p_id,
                        "kmer_vector": kmer_vec,
                        "k_size": k
                    })

                if len(batch) >= chunk_size:
                    dst_col.insert_many(batch, ordered=False)
                    inserted += len(batch)
                    pbar.update(len(batch))
                    batch = []

            if batch:
                dst_col.insert_many(batch, ordered=False)
                inserted += len(batch)
                pbar.update(len(batch))

    finally:
        cursor.close()

    print(f"[{time.strftime('%H:%M:%S')}] Indexando colección 'vec_kmers'...")
    dst_col.create_index("protein_id")
    print(f"[{time.strftime('%H:%M:%S')}] Proceso finalizado exitosamente. Insertados: {inserted:,} documentos.")


if __name__ == "__main__":
    run_kmers_vectorization_full(k=3, chunk_size=10000)