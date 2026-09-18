#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualizador Interactivo GPU-Acelerado (cuML / CUDA) con Clustering kNN + Leiden.

Utiliza RAPIDS cuML para SVD y UMAP en GPU, e igraph + leidenalg para la detección de comunidades  sobre 70,000+ muestras.
"""

import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning, module="cuda")

from cuda.bindings import runtime as cudart
from cuda.bindings import driver as cuda_driver

import numpy as np
from pymongo import MongoClient
import cupy as cp
from cuml.manifold import UMAP as cuUMAP
from cuml.neighbors import NearestNeighbors as cuNN
import cupyx.scipy.sparse as cps
from cupyx.scipy.sparse.linalg import svds as cupy_svds
from scipy.sparse import issparse, csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

import igraph as ig
import leidenalg as la

import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extract_vector(doc_field) -> np.ndarray:
    """Convierte listas o dicts indexados a arreglos NumPy de tipo float32."""
    if isinstance(doc_field, list):
        return np.array(doc_field, dtype=np.float32)
    elif isinstance(doc_field, dict):
        sorted_keys = sorted(doc_field.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        return np.array([doc_field[k] for k in sorted_keys], dtype=np.float32)
    return np.array([], dtype=np.float32)


def load_unified_data(db, sample_size: int = 70000):
    print(f"[{time.strftime('%H:%M:%S')}] Extrayendo documentos desde 'genes_unified_features'...")
    col = db["genes_unified_features"]
    
    cursor = col.find(
        {},
        {
            "_id": 0,
            "protein_id": 1,
            "product": 1,
            "kmers6_vector": 1,
            "prott5_vector": 1,
            "esm3_vector": 1
        }
    ).limit(sample_size * 2)

    docs = []
    for d in cursor:
        if d.get("protein_id") and d.get("product") and d.get("kmers6_vector") and d.get("prott5_vector") and d.get("esm3_vector"):
            docs.append(d)
        if len(docs) == sample_size:
            break

    print(f"[{time.strftime('%H:%M:%S')}] Documentos cargados válidos: {len(docs):,}")

    ids = [d["protein_id"] for d in docs]
    y_product = [d["product"] for d in docs]

    print(f"[{time.strftime('%H:%M:%S')}] Procesando k-mers como matriz esparcida (Sparse Float32)...")
    
    # 1. 6-Mers: Forzar float32 y NUNCA usar .toarray()
    if isinstance(docs[0]["kmers6_vector"], dict):
        vec_tool = DictVectorizer(sparse=True, dtype=np.float32)
        X_k6 = vec_tool.fit_transform([d["kmers6_vector"] for d in docs])
    else:
        X_k6 = np.array([extract_vector(d["kmers6_vector"]) for d in docs], dtype=np.float32)

    # 2. Embeddings pLLMs (Densos)
    print(f"[{time.strftime('%H:%M:%S')}] Procesando embeddings pLLMs...")
    X_pt5 = np.array([extract_vector(d["prott5_vector"]) for d in docs], dtype=np.float32)
    X_esm = np.array([extract_vector(d["esm3_vector"]) for d in docs], dtype=np.float32)

    return ids, y_product, X_k6, X_pt5, X_esm


def run_leiden_clustering(X_gpu_dense, n_neighbors: int = 15, resolution: float = 0.5) -> np.ndarray:
    """Construye un grafo k-NN con cuML y aplica la detección de comunidades de Leiden."""
    # 1. Calcular los k vecinos más cercanos en GPU
    nn = cuNN(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_gpu_dense)
    knn_graph_cupy = nn.kneighbors_graph(X_gpu_dense, mode='connectivity')
    
    # 2. Convertir la matriz de adyacencia a SciPy CSR en CPU
    knn_graph_cpu = csr_matrix(knn_graph_cupy.get())
    
    # 3. Crear el grafo igraph
    sources, targets = knn_graph_cpu.nonzero()
    edges = list(zip(sources, targets))
    g = ig.Graph(n=X_gpu_dense.shape[0], edges=edges, directed=False)
    g.simplify()  # Remover duplicados y auto-bucles
    
    # 4. Clustering con Leiden (Partition RBConfiguration)
    partition = la.find_partition(
        g, 
        la.RBConfigurationVertexPartition, 
        resolution_parameter=resolution,
        seed=42
    )
    
    return np.array(partition.membership)


def project_and_cluster_gpu(X, n_components: int, metric: str = "cosine", resolution: float = 0.5):
    """Procesa matrices densas/esparcidas usando SVD en GPU, UMAP y clustering kNN + Leiden."""
    
    if issparse(X):
        print(f"[{time.strftime('%H:%M:%S')}] Matriz esparcida detectada ({X.shape[1]:,} cols). Reduciendo con CuPy svds (GPU -> 50 cols)...")
        
        # 1. Cargar la matriz esparcida en VRAM como float32
        X_gpu_sparse = cps.csr_matrix(X, dtype=cp.float32)
        
        # 2. SVD esparcida acelerada en GPU (k=50)
        u, s, _ = cupy_svds(X_gpu_sparse, k=50)
        X_gpu = u * s
        
        # Liberar la matriz esparcida de VRAM
        del X_gpu_sparse, u, s
        cp.get_default_memory_pool().free_all_blocks()
        
        metric_to_use = "euclidean"
    else:
        # Normalización L2 para embeddings pLLMs densos
        X_norm = normalize(X, norm="l2", axis=1)
        X_gpu = cp.asarray(X_norm, dtype=cp.float32)
        metric_to_use = metric

    # 3. Reducción UMAP en VRAM
    reducer = cuUMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric=metric_to_use,
        random_state=42
    )
    coords_gpu = reducer.fit_transform(X_gpu)

    # 4. Clustering kNN + Leiden
    labels = run_leiden_clustering(X_gpu, n_neighbors=15, resolution=resolution)

    # Limpieza de VRAM
    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return cp.asnumpy(coords_gpu), labels


def generate_interactive_plots(ids, y_product, X_k6, X_pt5, X_esm):
    models = [
        ("vec_esm3", X_esm, "cosine"),
        ("vec_prott5", X_pt5, "cosine"),
        ("vec_kmers6", X_k6, "cosine")
    ]

    fig_2d = make_subplots(
        rows=1, cols=3,
        subplot_titles=("ESM-3 (2D)", "ProtT5 (2D)", "6-Mers (2D)"),
        horizontal_spacing=0.04
    )

    fig_3d = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("ESM-3 (3D)", "ProtT5 (3D)", "6-Mers (3D)"),
        horizontal_spacing=0.02
    )

    for idx, (name, X, metric) in enumerate(models, 1):
        print(f"[{time.strftime('%H:%M:%S')}] Ejecutando UMAP 2D + Leiden para {name}...")
        coords_2d, labels_2d = project_and_cluster_gpu(X, n_components=2, metric=metric, resolution=0.5)

        # En la traza 2D:
        fig_2d.add_trace(
            go.Scattergl(
                x=coords_2d[:, 0],
                y=coords_2d[:, 1],
                mode='markers',
                text=[f"ID: {p}<br>Prod: {prod}<br>Comunidad Leiden: {c}" for p, prod, c in zip(ids, y_product, labels_2d)],
                marker=dict(size=2, color=labels_2d, colorscale='Turbo', opacity=0.6),
                name=name
            ), row=1, col=idx
        )

        print(f"[{time.strftime('%H:%M:%S')}] Ejecutando UMAP 3D + Leiden para {name}...")
        coords_3d, labels_3d = project_and_cluster_gpu(X, n_components=3, metric=metric, resolution=0.5)

        # En la traza 3D:
        fig_3d.add_trace(
            go.Scatter3d(
                x=coords_3d[:, 0],
                y=coords_3d[:, 1],
                z=coords_3d[:, 2],
                mode='markers',
                text=[f"ID: {p}<br>Prod: {prod}<br>Comunidad Leiden: {c}" for p, prod, c in zip(ids, y_product, labels_3d)],
                hoverinfo='text',
                marker=dict(size=1.5, color=labels_3d, colorscale='Turbo', opacity=0.7),
                name=name
            ), row=1, col=idx
        )

    fig_2d.update_layout(
        title="Topología Espacial (UMAP 2D + kNN/Leiden)",
        template="plotly_dark",
        height=600,
        showlegend=False
    )
    fig_3d.update_layout(
        title="Espacios Latentes Tridimensionales (UMAP 3D + kNN/Leiden)",
        template="plotly_dark",
        height=750,
        showlegend=False
    )

    print(f"[{time.strftime('%H:%M:%S')}] Exportando archivos HTML...")
    fig_2d.write_html("visualizacion_leiden_2D.html")
    fig_3d.write_html("visualizacion_leiden_3D.html")
    print(f"[{time.strftime('%H:%M:%S')}] Proceso finalizado con éxito.")


def main():
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]
    ids, y_product, X_k6, X_pt5, X_esm = load_unified_data(db, sample_size=70000)
    generate_interactive_plots(ids, y_product, X_k6, X_pt5, X_esm)


if __name__ == "__main__":
    main()