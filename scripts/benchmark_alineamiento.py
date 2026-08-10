#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark Comparativo de Estrategias de Vectorización por Alineamiento.

Evalúa y compara las tres estrategias (Landmark MDS, MMseqs2 Multi-Threshold y 
CD-HIT Multi-Threshold) sobre una muestra de 70.000 registros usando UMAP + HDBSCAN
y calculando métricas intrínsecas (Silueta, Davies-Bouldin, Calinski-Harabasz) y 
extrínsecas (ARI, NMI, Homogeneidad, Completitud, V-Measure) frente a 'product'.
"""

import time
import numpy as np
import pandas as pd
from pymongo import MongoClient
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


def extract_vector(doc_field) -> np.ndarray:
    """Convierte campos tipo dict o list provenientes de MongoDB en arreglos NumPy unidimensionales."""
    if isinstance(doc_field, list):
        return np.array(doc_field, dtype=np.float32)
    elif isinstance(doc_field, dict):
        # Si las claves son ordinales o representaciones numéricas
        sorted_keys = sorted(doc_field.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        return np.array([doc_field[k] for k in sorted_keys], dtype=np.float32)
    return np.array([], dtype=np.float32)


def load_dataset_and_vectors(
    db, sample_size: int = 70000
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Carga 70,000 proteínas de 'genes_curados' y cruza sus identificadores con las BBDD de vectores."""
    print(f"[{time.strftime('%H:%M:%S')}] Extrayendo muestra de {sample_size:,} secuencias de 'genes_curados'...")
    src_col = db["genes_curados"]

    pipeline = [
        {
            "$match": {
                "product": {"$exists": True, "$ne": None, "$ne": ""},
                "protein_id": {"$exists": True, "$ne": None, "$ne": ""},
            }
        },
        {"$sample": {"size": sample_size}},
        {"$project": {"_id": 0, "protein_id": 1, "product": 1}},
    ]

    docs = list(src_col.aggregate(pipeline))
    if not docs:
        raise ValueError("No se encontraron registros válidos en 'genes_curados'.")

    protein_id_to_product = {d["protein_id"]: d["product"] for d in docs}
    sample_ids = set(protein_id_to_product.keys())

    print(f"[{time.strftime('%H:%M:%S')}] Obteniendo vectores de 'vec_align_mds', 'vec_mmseqs2' y 'vec_cdhit'...")

    # Cargar vectores de Landmark MDS
    mds_docs = db["vec_align_mds"].find(
        {"protein_id": {"$in": list(sample_ids)}},
        {"_id": 0, "protein_id": 1, "align_mds_vector": 1},
    )
    mds_map = {
        d["protein_id"]: extract_vector(d.get("align_mds_vector"))
        for d in mds_docs
        if "align_mds_vector" in d
    }

    # Cargar vectores de MMseqs2
    mmseqs_docs = db["vec_mmseqs2"].find(
        {"protein_id": {"$in": list(sample_ids)}},
        {"_id": 0, "protein_id": 1, "mmseq2_vector": 1, "mmseqs2_vector": 1},
    )
    mmseqs_map = {}
    for d in mmseqs_docs:
        vec_data = d.get("mmseq2_vector") or d.get("mmseqs2_vector")
        if vec_data is not None:
            mmseqs_map[d["protein_id"]] = extract_vector(vec_data)

    # Cargar vectores de CD-HIT
    cdhit_docs = db["vec_cdhit"].find(
        {"protein_id": {"$in": list(sample_ids)}},
        {"_id": 0, "protein_id": 1, "cdhit_vector": 1},
    )
    cdhit_map = {
        d["protein_id"]: extract_vector(d.get("cdhit_vector"))
        for d in cdhit_docs
        if "cdhit_vector" in d
    }

    # Intersección común de protein_ids
    common_ids = [
        p_id
        for p_id in sample_ids
        if p_id in mds_map and p_id in mmseqs_map and p_id in cdhit_map
    ]

    print(f"[{time.strftime('%H:%M:%S')}] Registros con cobertura completa en las 3 BBDD: {len(common_ids):,}")

    y_product = np.array([protein_id_to_product[p_id] for p_id in common_ids])
    
    X_dict = {
        "Landmark MDS": np.array([mds_map[p_id] for p_id in common_ids], dtype=np.float32),
        "MMseqs2": np.array([mmseqs_map[p_id] for p_id in common_ids], dtype=np.float32),
        "CD-HIT": np.array([cdhit_map[p_id] for p_id in common_ids], dtype=np.float32),
    }

    return y_product, X_dict


def run_umap_hdbscan(X: np.ndarray, metric: str = "cosine", min_cluster_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Aplica reducción UMAP a d=10 y realiza clustering HDBSCAN."""
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.1,
        metric=metric,
        random_state=42,
        low_memory=True,
    )
    X_reduced = reducer.fit_transform(X)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X_reduced)

    return labels, X_reduced


def compute_clustering_metrics(
    X_reduced: np.ndarray, labels: np.ndarray, y_true: np.ndarray
) -> dict[str, float]:
    """Calcula el conjunto completo de métricas intrínsecas y extrínsecas."""
    valid_mask = labels != -1
    n_clusters = len(np.unique(labels[valid_mask]))
    noise_ratio = round(float(np.sum(~valid_mask) / len(labels)), 4)

    # Métricas Extrínsecas (Supervisadas)
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
        "Calinski_Harabasz": np.nan,
    }

    # Métricas Intrínsecas (No supervisadas, sobre puntos sin ruido)
    if n_clusters > 1 and np.sum(valid_mask) > n_clusters:
        X_valid = X_reduced[valid_mask]
        labels_valid = labels[valid_mask]

        sil = silhouette_score(
            X_valid, labels_valid, metric="euclidean", sample_size=10000, random_state=42
        )
        db_idx = davies_bouldin_score(X_valid, labels_valid)
        ch_idx = calinski_harabasz_score(X_valid, labels_valid)

        metrics["Silueta"] = round(float(sil), 4)
        metrics["Davies_Bouldin"] = round(float(db_idx), 4)
        metrics["Calinski_Harabasz"] = round(float(ch_idx), 2)

    return metrics


def main():
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    y_product, X_dict = load_dataset_and_vectors(db, sample_size=70000)
    results = []

    # Configuración de métricas UMAP según tipo de vector
    metrics_umap = {
        "Landmark MDS": "cosine",     # Espacio continuo euclídeo
        "MMseqs2": "manhattan",       # Espacio categórico multiresolución
        "CD-HIT": "manhattan",        # Espacio categórico multiresolución
    }

    for model_name, X in X_dict.items():
        print(f"\n==================================================")
        print(f" EVALUANDO ESTRATEGIA DE ALINEAMIENTO: {model_name}")
        print(f"==================================================")
        
        umap_metric = metrics_umap[model_name]
        print(f"[{time.strftime('%H:%M:%S')}] Dimensiones de entrada: {X.shape}")
        
        labels, X_reduced = run_umap_hdbscan(X, metric=umap_metric, min_cluster_size=20)

        print(f"[{time.strftime('%H:%M:%S')}] Calculando métricas intrínsecas y extrínsecas...")
        metrics = compute_clustering_metrics(X_reduced, labels, y_product)

        res_entry = {
            "Modelo": model_name,
            "Dimensiones_Originales": X.shape[1],
            **metrics
        }
        results.append(res_entry)

    df_res = pd.DataFrame(results)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n\n" + "=" * 100)
    print(" BENCHMARK COMPARATIVO DE ESTRATEGIAS DE VECTORIZACIÓN POR ALINEAMIENTO")
    print("=" * 100)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()