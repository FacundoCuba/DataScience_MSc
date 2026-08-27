#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark de Feature Fusion acelerado por GPU (cuML / CuPy).

Alinea colecciones en MongoDB mediante $lookup en la base de datos,
evitando errores de consistencia en memoria durante el Early Fusion.
"""

import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from pymongo import MongoClient

import cupy as cp
import cupyx.scipy.sparse as cp_sparse
import cuml
from cuml.decomposition import TruncatedSVD as cumlTruncatedSVD
from cuml.manifold import UMAP as cumlUMAP
from cuml.cluster import HDBSCAN as cumlHDBSCAN
from cuml.neighbors import NearestNeighbors as cumlNearestNeighbors

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    homogeneity_completeness_v_measure,
)


def extract_dense_array(doc_field):
    """Convierte listas, arrays o diccionarios indexados de Mongo a np.ndarray 1D float32."""
    if doc_field is None:
        return None
    if isinstance(doc_field, (list, tuple, np.ndarray)):
        arr = np.array(doc_field, dtype=np.float32)
        return arr if arr.ndim == 1 and arr.size > 0 else None
    if isinstance(doc_field, dict):
        if not doc_field:
            return None
        try:
            max_idx = max(int(k) for k in doc_field.keys())
            arr = np.zeros(max_idx + 1, dtype=np.float32)
            for k, v in doc_field.items():
                arr[int(k)] = float(v)
            return arr
        except (ValueError, TypeError):
            return None
    return None


def fetch_aligned_fusion_sample(db, required_collections: list[str], sample_size: int = 70000):
    """Alinea e interseca colecciones directamente en MongoDB mediante pipeline de $lookup."""
    print(f"[{time.strftime('%H:%M:%S')}] Ejecutando $lookup en MongoDB para colecciones: {required_collections}")
    
    pipeline = [
        {"$match": {"product": {"$exists": True, "$ne": None, "$ne": ""}, "protein_id": {"$exists": True, "$ne": None, "$ne": ""}}},
        {"$sample": {"size": sample_size}}
    ]

    for col in required_collections:
        pipeline.extend([
            {
                "$lookup": {
                    "from": col,
                    "localField": "protein_id",
                    "foreignField": "protein_id",
                    "as": f"join_{col}"
                }
            },
            {"$unwind": f"$join_{col}"}
        ])

    # Corrección en la proyección: referenciar al alias aplanado por $unwind
    project_dict = {"_id": 0, "protein_id": 1, "product": 1}
    for col in required_collections:
        project_dict[col] = f"$join_{col}"

    pipeline.append({"$project": project_dict})

    docs = list(db["genes_curados"].aggregate(pipeline, allowDiskUse=True))
    if not docs:
        raise ValueError("No se encontraron documentos coincidentes en la intersección de colecciones.")

    print(f"[{time.strftime('%H:%M:%S')}] Registros perfectamente intersecados: {len(docs):,}")
    return docs


def process_feature_space(docs: list, col_name: str, candidates: list[str], is_kmer: bool, n_svd_components: int = 256):
    """Extrae y normaliza un espacio vectorial a partir de los documentos unificados."""
    extracted_data = []
    for d in docs:
        sub_doc = d.get(col_name, {})
        # Si sub_doc sigue siendo una lista (por alguna inconsistencia en el pipeline), tomar el primer elemento
        if isinstance(sub_doc, list) and len(sub_doc) > 0:
            sub_doc = sub_doc[0]

        val = None
        if isinstance(sub_doc, dict):
            for cand in candidates:
                if cand in sub_doc and sub_doc[cand] is not None:
                    val = sub_doc[cand]
                    break
        extracted_data.append(val)

    if is_kmer:
        # Lógica Dispersa (DictVectorizer) -> Conversión a GPU -> SVD en cuML
        clean_dicts = []
        for v in extracted_data:
            if isinstance(v, dict) and len(v) > 0:
                clean_dicts.append({str(k): float(val) for k, val in v.items() if isinstance(val, (int, float, np.number))})
            else:
                clean_dicts.append({})

        vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
        X_sparse_cpu = vectorizer.fit_transform(clean_dicts).tocsr()

        n_feats = X_sparse_cpu.shape[1]
        if n_feats == 0:
            raise ValueError(f"No se extrajeron features en la colección {col_name}.")

        actual_svd = min(n_svd_components, n_feats - 1) if n_feats > 1 else 1

        # FIX CLAVE: Convertir la matriz de SciPy (CPU) a CuPy Sparse CSR (GPU)
        X_sparse_gpu = cp_sparse.csr_matrix(X_sparse_cpu, dtype=cp.float32)

        svd = cumlTruncatedSVD(n_components=actual_svd, random_state=42)
        X_dense_gpu = svd.fit_transform(X_sparse_gpu)
        
        X_dense = cp.asnumpy(X_dense_gpu) if isinstance(X_dense_gpu, cp.ndarray) else np.asarray(X_dense_gpu)
        return normalize(X_dense.astype(np.float32), norm="l2", axis=1)
    else:
        # Lógica Densa (Arrays NumPy)
        vecs = [extract_dense_array(v) for v in extracted_data]
        first_valid = next((v for v in vecs if v is not None), None)
        if first_valid is None:
            raise ValueError(f"No se encontraron vectores densos válidos en la columna {col_name}")

        dim = first_valid.shape[0]
        clean_vecs = [
            v if (v is not None and isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == dim) 
            else np.zeros(dim, dtype=np.float32) 
            for v in vecs
        ]
        
        X = np.vstack(clean_vecs).astype(np.float32)
        return normalize(X, norm="l2", axis=1)


def run_umap_hdbscan(X, min_cluster_size=20):
    reducer = cumlUMAP(n_components=10, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    X_reduced_gpu = reducer.fit_transform(X)

    clusterer = cumlHDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    labels_gpu = clusterer.fit_predict(X_reduced_gpu)

    X_reduced = cp.asnumpy(X_reduced_gpu) if isinstance(X_reduced_gpu, cp.ndarray) else X_reduced_gpu
    labels = cp.asnumpy(labels_gpu) if isinstance(labels_gpu, cp.ndarray) else labels_gpu

    return labels, X_reduced


def run_knn_leiden(X_reduced, k_neighbors=15, resolution=0.005):
    nn = cumlNearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
    nn.fit(X_reduced)
    distances_gpu, indices_gpu = nn.kneighbors(X_reduced)

    distances = cp.asnumpy(distances_gpu)
    indices = cp.asnumpy(indices_gpu)

    n_samples = X_reduced.shape[0]
    sources = np.repeat(np.arange(n_samples), k_neighbors)
    targets = indices.ravel()
    weights = 1.0 / (1.0 + distances.ravel())

    g = ig.Graph(n=n_samples, edges=list(zip(sources, targets)), directed=False, edge_attrs={"weight": weights})
    g.simplify(combine_edges=max)

    partition = leidenalg.find_partition(
        g, leidenalg.CPMVertexPartition, weights="weight", resolution_parameter=resolution, seed=42
    )
    return np.array(partition.membership)


def compute_metrics(X_eval, labels, y_true):
    valid_mask = labels != -1
    n_clusters = len(np.unique(labels[valid_mask])) if np.any(valid_mask) else 0
    noise_ratio = round(float(np.sum(~valid_mask) / len(labels)), 4)

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
    }

    if n_clusters > 1 and np.sum(valid_mask) > n_clusters:
        X_valid = X_eval[valid_mask]
        labels_valid = labels[valid_mask]
        metrics["Silueta"] = round(float(silhouette_score(X_valid, labels_valid, metric="euclidean", sample_size=10000, random_state=42)), 4)
        metrics["Davies_Bouldin"] = round(float(davies_bouldin_score(X_valid, labels_valid)), 4)
        metrics["Calinski_Harabasz"] = round(float(calinski_harabasz_score(X_valid, labels_valid)), 2)
    else:
        metrics["Silueta"], metrics["Davies_Bouldin"], metrics["Calinski_Harabasz"] = np.nan, np.nan, np.nan

    return metrics


def main():
    gpu_id = cp.cuda.Device().id
    gpu_name = cp.cuda.runtime.getDeviceProperties(gpu_id)['name'].decode('utf-8')
    print(f"[-] Inicializando benchmark sobre GPU {gpu_id}: {gpu_name}")

    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    vector_configs = {
        "kmers6": ("vec_kmers6", ["kmer_vector", "kmer6_vector", "kmers6_vector", "vector"], True),
        "align_mds": ("vec_align_mds", ["align_mds_vector", "vector"], False),
        "esm3": ("vec_esm3", ["esm3_vector", "vector"], False),
        "prott5": ("vec_prott5", ["prott5_vector", "vector"], False),
    }

    combinations = [
        ("kmers6 + esm3", [vector_configs["kmers6"], vector_configs["esm3"]]),
        ("kmers6 + prott5", [vector_configs["kmers6"], vector_configs["prott5"]]),
        ("kmers6 + align_mds", [vector_configs["kmers6"], vector_configs["align_mds"]]),
        ("align_mds + esm3", [vector_configs["align_mds"], vector_configs["esm3"]]),
        ("align_mds + prott5", [vector_configs["align_mds"], vector_configs["prott5"]]),
        ("kmers6 + align_mds + esm3", [vector_configs["kmers6"], vector_configs["align_mds"], vector_configs["esm3"]]),
        ("kmers6 + align_mds + prott5", [vector_configs["kmers6"], vector_configs["align_mds"], vector_configs["prott5"]]),
    ]

    results = []

    for combo_name, combo_configs in combinations:
        print(f"\n==================================================")
        print(f" EVALUANDO COMBINACIÓN: {combo_name}")
        print(f"==================================================")

        try:
            req_cols = [cfg[0] for cfg in combo_configs]
            docs = fetch_aligned_fusion_sample(db, req_cols, sample_size=70000)
            
            y_fused = np.array([d["product"] for d in docs])
            
            matrices = []
            for col_name, candidates, is_kmer in combo_configs:
                X_part = process_feature_space(docs, col_name, candidates, is_kmer=is_kmer)
                print(f"[{time.strftime('%H:%M:%S')}] -> Subespacio '{col_name}' procesado con éxito. Shape: {X_part.shape}")
                matrices.append(X_part)

            X_fused = np.hstack(matrices)
            
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR en la combinación {combo_name}: {e}")
            continue

        print(f"[{time.strftime('%H:%M:%S')}] Muestra efectiva: {X_fused.shape[0]:,} registros")
        print(f"[{time.strftime('%H:%M:%S')}] Dimensiones fusionadas: {X_fused.shape[1]:,}")

        # Pipeline 1: UMAP + HDBSCAN (cuML en GPU)
        t0 = time.time()
        labels_hdb, X_red = run_umap_hdbscan(X_fused, min_cluster_size=20)
        m_hdb = compute_metrics(X_red, labels_hdb, y_fused)
        print(f"[{time.strftime('%H:%M:%S')}] UMAP+HDBSCAN completado en {time.time() - t0:.2f}s")
        results.append({"Combinacion": combo_name, "Algoritmo": "UMAP+HDBSCAN", "Dims": X_fused.shape[1], **m_hdb})

        # Pipeline 2: kNN (cuML) + Leiden
        t0 = time.time()
        labels_lei = run_knn_leiden(X_red, k_neighbors=15, resolution=0.005)
        m_lei = compute_metrics(X_red, labels_lei, y_fused)
        print(f"[{time.strftime('%H:%M:%S')}] kNN+Leiden completado en {time.time() - t0:.2f}s")
        results.append({"Combinacion": combo_name, "Algoritmo": "kNN+Leiden", "Dims": X_fused.shape[1], **m_lei})

    df_res = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + "=" * 120)
    print(" REPORTE DE COMBINACIONES MÚLTIPLES (FEATURE FUSION)")
    print("=" * 120)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()