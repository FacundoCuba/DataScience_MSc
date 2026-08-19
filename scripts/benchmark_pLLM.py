#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark Comparativo de Estrategias de Vectorización por pLLMs.

Evalúa y compara las estrategias de embeddings basadas en Protein Language Models 
(ESM-2, ESM-3 y ProtT5) sobre una muestra de 70.000 registros mediante 
UMAP + HDBSCAN, calculando métricas intrínsecas (Silueta, Davies-Bouldin, Calinski-Harabasz) 
y extrínsecas (ARI, NMI, Homogeneidad, Completitud, V-Measure) frente a 'product'.
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
        sorted_keys = sorted(doc_field.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        return np.array([doc_field[k] for k in sorted_keys], dtype=np.float32)
    return np.array([], dtype=np.float32)


def load_dataset_and_vectors(
    db, sample_size: int = 70000
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Carga 70,000 proteínas de 'genes_curados' y cruza sus identificadores con las BBDD de vectores pLLM."""
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

    print(f"[{time.strftime('%H:%M:%S')}] Obteniendo vectores de 'vec_esm2', 'vec_esm3' y 'vec_prott5'...")

    # Cargar vectores ESM-2
    esm2_docs = db["vec_esm2"].find(
        {"protein_id": {"$in": list(sample_ids)}},
        {"_id": 0, "protein_id": 1, "esm2_vector": 1, "vector": 1},
    )
    esm2_map = {}
    for d in esm2_docs:
        vec_data = d.get("esm2_vector") if "esm2_vector" in d else d.get("vector")
        if vec_data is not None:
            esm2_map[d["protein_id"]] = extract_vector(vec_data)

    # Cargar vectores ESM-3
    esm3_docs = db["vec_esm3"].find(
        {"protein_id": {"$in": list(sample_ids)}},
        {"_id": 0, "protein_id": 1, "esm3_vector": 1, "vector": 1},
    )
    esm3_map = {}
    for d in esm3_docs:
        vec_data = d.get("esm3_vector") if "esm3_vector" in d else d.get("vector")
        if vec_data is not None:
            esm3_map[d["protein_id"]] = extract_vector(vec_data)

    # Cargar vectores ProtT5
    prott5_docs = db["vec_prott5"].find(
        {"protein_id": {"$in": list(sample_ids)}},
        {"_id": 0, "protein_id": 1, "prott5_vector": 1, "vector": 1},
    )
    prott5_map = {}
    for d in prott5_docs:
        vec_data = d.get("prott5_vector") if "prott5_vector" in d else d.get("vector")
        if vec_data is not None:
            prott5_map[d["protein_id"]] = extract_vector(vec_data)

    # Intersección común de protein_ids
    common_ids = [
        p_id
        for p_id in sample_ids
        if p_id in esm2_map and p_id in esm3_map and p_id in prott5_map
    ]

    print(f"[{time.strftime('%H:%M:%S')}] Registros con cobertura completa en las 3 BBDD de pLLMs: {len(common_ids):,}")

    if not common_ids:
        raise ValueError("No se encontraron registros en común entre las 3 colecciones de embeddings.")

    y_product = np.array([protein_id_to_product[p_id] for p_id in common_ids])

    X_dict = {
        "ESM-2": np.array([esm2_map[p_id] for p_id in common_ids], dtype=np.float32),
        "ESM-3": np.array([esm3_map[p_id] for p_id in common_ids], dtype=np.float32),
        "ProtT5": np.array([prott5_map[p_id] for p_id in common_ids], dtype=np.float32),
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

    # Configuración de métricas UMAP: para embeddings latentes de pLLMs se utiliza la distancia coseno
    metrics_umap = {
        "ESM-2": "cosine",
        "ESM-3": "cosine",
        "ProtT5": "cosine",
    }

    for model_name, X in X_dict.items():
        print(f"\n==================================================")
        print(f" EVALUANDO ESTRATEGIA pLLM: {model_name}")
        print(f"==================================================")

        umap_metric = metrics_umap[model_name]
        print(f"[{time.strftime('%H:%M:%S')}] Dimensiones de entrada: {X.shape}")

        labels, X_reduced = run_umap_hdbscan(X, metric=umap_metric, min_cluster_size=20)

        print(f"[{time.strftime('%H:%M:%S')}] Calculando métricas intrínsecas y extrínsecas...")
        metrics = compute_clustering_metrics(X_reduced, labels, y_product)

        res_entry = {
            "Modelo": model_name,
            "Dimensiones_Originales": X.shape[1],
            **metrics,
        }
        results.append(res_entry)

    df_res = pd.DataFrame(results)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n\n" + "=" * 100)
    print(" BENCHMARK COMPARATIVO DE ESTRATEGIAS DE VECTORIZACIÓN POR pLLMs")
    print("=" * 100)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()