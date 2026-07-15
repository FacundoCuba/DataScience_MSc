# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import h5py
from pymongo import MongoClient
import cudf
import cuml
from cuml.cluster import HDBSCAN as cumlHDBSCAN
from cuml.manifold import UMAP as cumlUMAP
from sklearn.metrics import adjusted_rand_score, silhouette_score
import cupy as cp
import time

def benchmark_umap_evolucionado():
    print(f"[{time.strftime('%H:%M:%S')}] 1. Iniciando Benchmark con UMAP en GPU...")
    client = MongoClient('mongodb://localhost:27017/')
    db = client['viromica_db']
    col = db['genes_curados']

    # 2. Cargar IDs y Metadatos
    print(f"[{time.strftime('%H:%M:%S')}] 2. Cargar IDs y Metadatos...")
    cursor = col.find({}, {
        "protein_id": 1, 
        "metadata_virus.taxonomy": 1, 
        "cdhit_cluster_90": 1, 
        "cluster_mmseqs_90": 1
    })
    df_meta = pd.DataFrame(list(cursor))
    df_meta['familia_real'] = df_meta['metadata_virus'].apply(
        lambda x: x['taxonomy'][-2] if len(x['taxonomy']) > 1 else x['taxonomy'][-1]
    )

    # 3. Carga y Normalización de Embeddings
    print(f"[{time.strftime('%H:%M:%S')}] 3. Cargando y normalizando embeddings...")
    with h5py.File("embeddings_esm2.h5", "r") as h5f:
        emb_gpu = cp.array(h5f["vectors"][:])
    
    # Normalización L2 (crucial para embeddings de lenguaje)
    norm = cp.linalg.norm(emb_gpu, axis=1, keepdims=True)
    emb_gpu = emb_gpu / norm

    # 4. UMAP (Reducción No Lineal)
    print(f"[{time.strftime('%H:%M:%S')}] 4. Ejecutando UMAP (esto optimiza el espacio para HDBSCAN)...")
    reducer = cumlUMAP(
        n_neighbors=30, 
        n_components=10, 
        min_dist=0.1, 
        metric='euclidean',
        random_state=42
    )
    emb_reduced = reducer.fit_transform(emb_gpu)

    # 5. Clustering HDBSCAN
    print(f"[{time.strftime('%H:%M:%S')}] 5. Ejecutando HDBSCAN sobre proyeccion UMAP...")
    hdbscan = cumlHDBSCAN(
        min_cluster_size=10, 
        min_samples=5, 
        prediction_data=True
    )
    labels_hdbscan_esm = hdbscan.fit_predict(emb_reduced).get()

    # 6. Métricas
    print(f"[{time.strftime('%H:%M:%S')}] 6. Calculando metricas finales...")
    
    ari_esm = adjusted_rand_score(df_meta['familia_real'], labels_hdbscan_esm)
    ari_cdhit = adjusted_rand_score(df_meta['familia_real'], df_meta['cdhit_cluster_90'])
    
    # Evaluar cuánto ruido generó HDBSCAN (etiqueta -1)
    n_ruido = np.sum(labels_hdbscan_esm == -1)
    pct_ruido = (n_ruido / len(labels_hdbscan_esm)) * 100

    print("\n" + "="*60)
    print(f"RESULTADOS CON UMAP + HDBSCAN")
    print("="*60)
    print(f"ARI ESM-2 vs Taxonomia:  {ari_esm:.4f}")
    print(f"ARI CD-HIT vs Taxonomia: {ari_cdhit:.4f}")
    print(f"Porcentaje de ruido (unclustered): {pct_ruido:.2f}%")
    print("="*60)

    # Guardar resultados
    df_meta['esm_umap_hdbscan'] = labels_hdbscan_esm
    df_meta.to_csv("master_clustering_umap_3.csv", index=False)

if __name__ == "__main__":
    benchmark_umap_evolucionado()