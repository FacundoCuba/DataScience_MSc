#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorización Masiva de Secuencias Proteicas mediante Estadística de k-Mers.

Procesa el universo completo de secuencias contenidas en la colección 'genes_curados' 
(690,579 documentos) y persiste las representaciones dispersas basadas en la frecuencia 
relativa de $k$-mers en dos colecciones independientes de MongoDB: 'vec_kmers5' y 'vec_kmers6'.

Flujo de trabajo:
-----------------
1. Lectura por lotes (*batching*) mediante un cursor persistente (`no_cursor_timeout=True`).
2. Generación simultánea de perfiles de frecuencia para $k=5$ y $k=6$ por cada secuencia
   en una sola pasada de lectura para optimizar I/O.
3. Persistencia por bloques (`insert_many`) en las colecciones de destino 'vec_kmers5' y 'vec_kmers6'.
4. Creación automatizada de índices sobre `protein_id` en ambas colecciones.
"""

from collections import Counter
from pymongo import MongoClient
from tqdm import tqdm
import time

def extract_kmers(sequence: str, k: int) -> dict[str, float]:
    """Calcula el perfil de frecuencias relativas de $k$-mers para una secuencia proteica.

    Extrae todas las subsucesiones continuas de longitud `k` utilizando una ventana deslizante.
    Mapea el resultado en un diccionario esparcido que asigna a cada $k$-mer observado su 
    frecuencia relativa normalizada por la cantidad total de subcadenas extraíbles ($N - k + 1$).
    """
    n = len(sequence)
    if not sequence or n < k:
        return {}
    
    total_kmers = n - k + 1
    kmers_counts = Counter(sequence[i:i+k] for i in range(total_kmers))
    return {kmer: count / total_kmers for kmer, count in kmers_counts.items()}

def run_kmers_vectorization_dual(k_list: list[int] = [5, 6], chunk_size: int = 10000) -> None:
    """Ejecuta el pipeline masivo de vectorización para k=5 y k=6 sobre 'genes_curados'.

    Lee la colección origen una sola vez y genera los documentos correspondientes para 
    las colecciones 'vec_kmers5' y 'vec_kmers6', insertando por lotes para maximizar la eficiencia.
    """
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]
    src_col = db["genes_curados"]
    
    # Mapeo dinámico de colecciones destino según k
    dst_cols = {k: db[f"vec_kmers{k}"] for k in k_list}

    total_docs = src_col.count_documents({})
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando vectorización dual por k-mers (k={k_list}). Total genes: {total_docs:,}")
    
    # Reiniciamos las colecciones destino
    for k, col in dst_cols.items():
        col.drop()

    # Cursor robusto para evitar desconexiones en operaciones largas
    cursor = src_col.find(
        {}, 
        {"protein_id": 1, "aa_sequence": 1, "_id": 0},
        no_cursor_timeout=True
    ).batch_size(chunk_size)

    # Batches independientes por cada k
    batches = {k: [] for k in k_list}
    inserted_counts = {k: 0 for k in k_list}

    try:
        with tqdm(total=total_docs, desc="Procesando k=5 y k=6", unit="seq") as pbar:
            for doc in cursor:
                p_id = doc.get("protein_id")
                seq = doc.get("aa_sequence", "")
                
                if p_id and seq:
                    for k in k_list:
                        kmer_vec = extract_kmers(seq, k=k)
                        batches[k].append({
                            "protein_id": p_id,
                            "kmer_vector": kmer_vec,
                            "k_size": k
                        })

                # Inserción por bloques cuando se alcanza el tamaño de batch
                if len(batches[k_list[0]]) >= chunk_size:
                    for k in k_list:
                        dst_cols[k].insert_many(batches[k], ordered=False)
                        inserted_counts[k] += len(batches[k])
                        batches[k] = []
                    pbar.update(chunk_size)

            # Inserción del remanente final
            if batches[k_list[0]]:
                rem_len = len(batches[k_list[0]])
                for k in k_list:
                    dst_cols[k].insert_many(batches[k], ordered=False)
                    inserted_counts[k] += len(batches[k])
                    batches[k] = []
                pbar.update(rem_len)

    finally:
        cursor.close()

    # Creación de índices en ambas colecciones
    for k, col in dst_cols.items():
        print(f"[{time.strftime('%H:%M:%S')}] Indexando colección 'vec_kmers{k}' por protein_id...")
        col.create_index("protein_id")
        print(f"[{time.strftime('%H:%M:%S')}] Colección 'vec_kmers{k}' lista. Insertados: {inserted_counts[k]:,} documentos.")

if __name__ == "__main__":
    run_kmers_vectorization_dual(k_list=[5, 6], chunk_size=10000)