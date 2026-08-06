#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorización Masiva de Secuencias Proteicas mediante Estadística de k-Mers.

Procesa el universo completo de secuencias contenidas en la colección 'genes_curados' 
(690,579 documentos) y persiste las representaciones dispersas basadas en la frecuencia 
relativa de $k$-mers en la colección 'vec_kmers' de MongoDB.

Flujo de trabajo:
-----------------
1. Lectura por lotes (*batching*) mediante un cursor persistente (`no_cursor_timeout=True`)
   para gestionar de manera eficiente el volumen masivo de datos sin agotar el timeout de MongoDB.
2. Extracción de $k$-mers ($k=5$ por defecto) mediante una ventana deslizante de paso 1 
   y normalización por longitud de secuencia.
3. Persistencia por bloques (`insert_many`) en la colección `vec_kmers`, almacenando 
   el identificador único (`protein_id`), el diccionario de frecuencias dispersas y el tamaño $k$.
4. Creación automatizada de índices sobre `protein_id` en la colección destino para optimizar
   las búsquedas y cruces en etapas posteriores del pipeline.
"""

from collections import Counter
from pymongo import MongoClient
from tqdm import tqdm
import time

def extract_kmers(sequence: str, k: int = 5) -> dict[str, float]:
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

def run_kmers_vectorization_full(k: int = 5, chunk_size: int = 10000) -> None:
    """Ejecuta el pipeline masivo de vectorización por $k$-mers sobre la colección 'genes_curados'.

    Limpia la colección de destino 'vec_kmers', lee los datos mediante un cursor por bloques 
    e inserta los vectores resultantes por lotes (`insert_many`) supervisando el progreso 
    con `tqdm`. Al finalizar la inserción, genera un índice secundario sobre `protein_id`.
    """
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
    run_kmers_vectorization_full(k=5, chunk_size=10000)