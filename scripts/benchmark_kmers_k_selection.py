#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark de Evaluación de k-Mers Aminoacídicos para Anotación Funcional Proteica.

Este script evalúa la capacidad de las representaciones basadas en frecuencia de $k$-mers
(de $k=3$ a $k=8$) para agrupar secuencias proteicas virales según su función biológica ('product').
Permite determinar el límite de resolución de los métodos de conteo exacto frente a modelos 
de aprendizaje profundo (p. ej., ESM-2).

Flujo de trabajo:
-----------------
1. Extracción de una muestra representativa de secuencias proteicas (`aa_sequence`) y su 
   anotación funcional (`product`) desde MongoDB (`viromica_db.genes_curados`).
2. Vectorización dinámica al vuelo en matrices dispersas (`scipy.sparse.csr_matrix`) utilizando
   la frecuencia relativa de cada $k$-mer por secuencia.
3. Reducción de dimensionalidad no lineal mediante UMAP ($d=10$, distancia coseno) directamente 
   sobre las matrices dispersas.
4. Agrupamiento basado en densidad con HDBSCAN (`min_cluster_size=20`).
5. Cálculo de métricas de desempeño respecto al Ground Truth funcional: NMI (Normalized Mutual 
   Information) y ARI (Adjusted Rand Index), identificando además la proporción de ruido (-1).
"""

from collections import Counter
import time
import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
import umap
from hdbscan import HDBSCAN

def load_sequences_and_products(db, sample_size: int = 70000) -> tuple[list[str], list[str]]:
    """Carga una muestra aleatoria de secuencias aminoacídicas y sus anotaciones funcionales desde MongoDB.

    Filtra los documentos de la colección 'genes_curados' asegurando que existan cadenas válidas
    y no vacías tanto para la secuencia traducida ('aa_sequence') como para el producto proteico ('product').
    """
    print(f"[{time.strftime('%H:%M:%S')}] Cargando muestra de {sample_size:,} secuencias proteicas...")
    src_col = db["genes_curados"]

    pipeline = [
        {
            "$match": {
                "product": {"$exists": True, "$ne": None, "$ne": ""},
                "aa_sequence": {"$exists": True, "$ne": None, "$ne": ""}
            }
        },
        {"$sample": {"size": sample_size}},
        {"$project": {"_id": 0, "translation": "$aa_sequence", "product": 1}}
    ]
    
    docs = list(src_col.aggregate(pipeline))
    
    if not docs:
        raise ValueError(
            "No se encontraron documentos en 'genes_curados' con 'product' y 'aa_sequence' válidos."
        )

    sequences = [d["translation"] for d in docs if isinstance(d.get("translation"), str) and len(d["translation"]) > 0]
    products = [d["product"] for d in docs if isinstance(d.get("translation"), str) and len(d["translation"]) > 0]

    lengths = [len(s) for s in sequences]
    print(f"[{time.strftime('%H:%M:%S')}] Secuencias cargadas: {len(sequences):,}. "
          f"Largo promedio: {np.mean(lengths):.1f} aa (Min: {np.min(lengths)}, Max: {np.max(lengths)})")
    
    return sequences, products

def extract_kmers_sparse(sequences: list[str], k: int) -> csr_matrix:
    """Construye una matriz dispersa de frecuencias relativas de $k$-mers a partir de secuencias proteicas.

    Extrae subsucesiones de longitud `k` mediante una ventana deslizante de paso 1. Normaliza 
    los conteos dividiendo por el total de $k$-mers de cada secuencia (frecuencia relativa) para 
    mitigar el efecto de las variaciones en la longitud de las proteínas.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Generando conteo de {k}-mers para {len(sequences):,} secuencias...")
    start_t = time.time()
    
    vocab = {}
    vocab_counter = 0
    rows, cols, data = [], [], []

    for seq_idx, seq in enumerate(sequences):
        if len(seq) < k:
            continue
        
        # Extraer k-mers
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        kmer_counts = Counter(kmers)
        total_kmers = len(kmers)
        
        for kmer, count in kmer_counts.items():
            if kmer not in vocab:
                vocab[kmer] = vocab_counter
                vocab_counter += 1
            
            kmer_id = vocab[kmer]
            rows.append(seq_idx)
            cols.append(kmer_id)
            data.append(count / total_kmers) # Frecuencia relativa dentro de la proteína

    X_sparse = csr_matrix((data, (rows, cols)), shape=(len(sequences), len(vocab)), dtype=np.float32)
    
    print(f"[{time.strftime('%H:%M:%S')}] {k}-mers extraídos en {time.time() - start_t:.1f}s. "
          f"Dimensiones del vocabulario único observado: {len(vocab):,}")
    
    return X_sparse

def run_umap_hdbscan_sparse(X_sparse: csr_matrix, k_val: int, min_cluster_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Aplica reducción UMAP sobre una matriz dispersa de $k$-mers y agrupa por densidad con HDBSCAN.

    Aprovecha el soporte nativo de UMAP para matrices `csr_matrix` utilizando la métrica de 
    distancia Coseno en el espacio de alta dimensión, proyectando a $d=10$ dimensiones antes 
    de ejecutar HDBSCAN con la técnica de selección de clusters EOM (Excess of Mass).
    """
    print(f"[{time.strftime('%H:%M:%S')}] Reduciendo dimensionalidad con UMAP (métrica='cosine', d=10) para k={k_val}...")
    
    # UMAP soporta directamente scipy.sparse.csr_matrix con metrica coseno
    reducer = umap.UMAP(
        n_components=10, 
        n_neighbors=15, 
        min_dist=0.1, 
        metric="cosine", 
        random_state=42, 
        low_memory=True
    )
    X_reduced = reducer.fit_transform(X_sparse)
    
    print(f"[{time.strftime('%H:%M:%S')}] Ejecutando HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(X_reduced)
    
    return labels, X_reduced

def main():
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    # Muestra para la prueba de k-mers
    sequences, products = load_sequences_and_products(db, sample_size=70000)
    y_product = np.array(products)

    k_values = [3, 4, 5, 6, 7, 8]
    results = []

    for k in k_values:
        print(f"\n==================================================")
        print(f" EVALUANDO K-MERS CON K = {k}")
        print(f"==================================================")
        
        # 1. Extraer matriz esparcida
        X_sparse = extract_kmers_sparse(sequences, k=k)
        
        # 2. Agrupar con UMAP (Coseno) + HDBSCAN
        labels, X_reduced = run_umap_hdbscan_sparse(X_sparse, k_val=k, min_cluster_size=20)
        
        # 3. Métricas
        valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
        n_clusters = len(np.unique(labels[valid_mask]))
        noise_ratio = round(float(np.sum(~valid_mask) / len(labels)), 4) if -1 in labels else 0.0
        
        nmi = round(normalized_mutual_info_score(y_product, labels), 4)
        ari = round(adjusted_rand_score(y_product, labels), 4)

        results.append({
            "Tamaño_Kmer": f"{k}-mer",
            "Vocabulario_Observado": f"{X_sparse.shape[1]:,}",
            "Clusters_Detectados": n_clusters,
            "Proporcion_Ruido": noise_ratio,
            "NMI_Product": nmi,
            "ARI_Product": ari
        })

    df_res = pd.DataFrame(results)
    print("\n\n" + "="*80)
    print(" COMPARATIVA DE RESOLUCIÓN POR TAMAÑO DE K-MER")
    print("="*80)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()