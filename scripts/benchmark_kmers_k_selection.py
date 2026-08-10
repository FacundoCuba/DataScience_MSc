#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark de Evaluación de k-Mers Aminoacídicos para Anotación Funcional Proteica.

Este script evalúa la capacidad de las representaciones basadas en frecuencia de $k$-mers
(de $k=3$ a $k=8$) para agrupar secuencias proteicas virales según su función biológica ('product').

Métricas evaluadas:
-------------------
1. Intrínsecas (No supervisadas - Cohesión y Separación en espacio UMAP):
   - Coeficiente de Silueta (Silhouette Score)
   - Índice Davies-Bouldin (DBI)
   - Índice Calinski-Harabasz (CHI)
2. Extrínsecas (Supervisadas - Comparación con Ground Truth 'product'):
   - Índice Rand Ajustado (ARI)
   - Información Mutua Normalizada (NMI)
   - Homogeneidad (Homogeneity)
   - Completitud (Completeness)
   - Medida V (V-Measure)
"""

from collections import Counter
import time
import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    homogeneity_completeness_v_measure,
)
import umap
from hdbscan import HDBSCAN


def load_sequences_and_products(db, sample_size: int = 70000) -> tuple[list[str], list[str]]:
    """Carga una muestra aleatoria de secuencias aminoacídicas y sus anotaciones funcionales desde MongoDB."""
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
    """Construye una matriz dispersa de frecuencias relativas de $k$-mers a partir de secuencias proteicas."""
    print(f"[{time.strftime('%H:%M:%S')}] Generando conteo de {k}-mers para {len(sequences):,} secuencias...")
    start_t = time.time()

    vocab = {}
    vocab_counter = 0
    rows, cols, data = [], [], []

    for seq_idx, seq in enumerate(sequences):
        if len(seq) < k:
            continue

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
            data.append(count / total_kmers)

    X_sparse = csr_matrix((data, (rows, cols)), shape=(len(sequences), len(vocab)), dtype=np.float32)

    print(f"[{time.strftime('%H:%M:%S')}] {k}-mers extraídos en {time.time() - start_t:.1f}s. "
          f"Dimensiones del vocabulario único observado: {len(vocab):,}")

    return X_sparse


def run_umap_hdbscan_sparse(X_sparse: csr_matrix, k_val: int, min_cluster_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Aplica reducción UMAP sobre una matriz dispersa y agrupa por densidad con HDBSCAN."""
    print(f"[{time.strftime('%H:%M:%S')}] Reduciendo dimensionalidad con UMAP (métrica='cosine', d=10) para k={k_val}...")

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


def compute_clustering_metrics(X_reduced: np.ndarray, labels: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Calcula el conjunto completo de métricas intrínsecas y extrínsecas."""
    valid_mask = labels != -1
    n_clusters = len(np.unique(labels[valid_mask]))
    noise_ratio = round(float(np.sum(~valid_mask) / len(labels)), 4)

    # 1. Métricas Extrínsecas (Supervisadas)
    nmi = round(float(normalized_mutual_info_score(y_true, labels)), 4)
    ari = round(float(adjusted_rand_score(y_true, labels)), 4)
    homo, comp, v_meas = homogeneity_completeness_v_measure(y_true, labels)

    metrics = {
        "Clusters": n_clusters,
        "Ruido_Ratio": noise_ratio,
        "ARI": ari,
        "NMI": nmi,
        "Homogeneidad": round(float(homo), 4),
        "Completitud": round(float(comp), 4),
        "V_Measure": round(float(v_meas), 4),
        "Silueta": np.nan,
        "Davies_Bouldin": np.nan,
        "Calinski_Harabasz": np.nan
    }

    # 2. Métricas Intrínsecas (No supervisadas, evaluadas sobre puntos sin ruido)
    if n_clusters > 1 and np.sum(valid_mask) > n_clusters:
        X_valid = X_reduced[valid_mask]
        labels_valid = labels[valid_mask]

        # Silueta con submuestreo para escalabilidad
        sil = silhouette_score(X_valid, labels_valid, metric="euclidean", sample_size=10000, random_state=42)
        db_idx = davies_bouldin_score(X_valid, labels_valid)
        ch_idx = calinski_harabasz_score(X_valid, labels_valid)

        metrics["Silueta"] = round(float(sil), 4)
        metrics["Davies_Bouldin"] = round(float(db_idx), 4)
        metrics["Calinski_Harabasz"] = round(float(ch_idx), 2)

    return metrics


def main():
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    sequences, products = load_sequences_and_products(db, sample_size=70000)
    y_product = np.array(products)

    k_values = [3, 4, 5, 6, 7, 8]
    results = []

    for k in k_values:
        print(f"\n==================================================")
        print(f" EVALUANDO K-MERS CON K = {k}")
        print(f"==================================================")

        X_sparse = extract_kmers_sparse(sequences, k=k)
        labels, X_reduced = run_umap_hdbscan_sparse(X_sparse, k_val=k, min_cluster_size=20)

        print(f"[{time.strftime('%H:%M:%S')}] Calculando métricas intrínsecas y extrínsecas...")
        metrics = compute_clustering_metrics(X_reduced, labels, y_product)

        res_entry = {
            "Kmer": f"{k}-mer",
            "Vocabulario": f"{X_sparse.shape[1]:,}",
            **metrics
        }
        results.append(res_entry)

    df_res = pd.DataFrame(results)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n\n" + "="*100)
    print(" COMPARATIVA DE RESOLUCIÓN Y EVALUACIÓN DE CLÚSTERING POR TAMAÑO DE K-MER")
    print("="*100)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()