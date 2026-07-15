# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import h5py
from pymongo import MongoClient
import cuml
from cuml.cluster import HDBSCAN as cumlHDBSCAN
from cuml.manifold import UMAP as cumlUMAP
from sklearn.metrics import adjusted_rand_score
import cupy as cp
import time

def benchmark_agnostico():
    print(f"[{time.strftime('%H:%M:%S')}] 1. Iniciando Benchmark no supervisado...")
    client = MongoClient('mongodb://localhost:27017/')
    db = client['viromica_db']
    col = db['genes_curados']

    # Recuperar metadatos (solo los IDs de los clusters clasicos)
    cursor = col.find({}, {"protein_id": 1, "cdhit_cluster_90": 1, "cluster_mmseqs_90": 1})
    df_meta = pd.DataFrame(list(cursor))

    # Cargar y Normalizar Embeddings
    with h5py.File("embeddings_esm2.h5", "r") as h5f:
        emb_gpu = cp.array(h5f["vectors"][:])
    emb_gpu = emb_gpu / cp.linalg.norm(emb_gpu, axis=1, keepdims=True)

    # UMAP para estructurar el espacio latente
    print(f"[{time.strftime('%H:%M:%S')}] 2. Proyectando espacio latente con UMAP...")
    reducer = cumlUMAP(n_neighbors=30, n_components=10, min_dist=0.0, random_state=42)
    emb_reduced = reducer.fit_transform(emb_gpu)

    # Clustering por Densidad (ESM-2)
    print(f"[{time.strftime('%H:%M:%S')}] 3. Ejecutando HDBSCAN...")
    hdbscan = cumlHDBSCAN(min_cluster_size=15, min_samples=5)
    labels_esm = hdbscan.fit_predict(emb_reduced).get()

    # Calculo de concordancia entre metodos (ARI)
    ari_esm_vs_cdhit = adjusted_rand_score(labels_esm, df_meta['cdhit_cluster_90'])
    ari_esm_vs_mmseqs = adjusted_rand_score(labels_esm, df_meta['cluster_mmseqs_90'])
    ari_classics = adjusted_rand_score(df_meta['cdhit_cluster_90'], df_meta['cluster_mmseqs_90'])

    print("\n" + "="*60)
    print(f"MATRIZ DE CONCORDANCIA ENTRE ESTRATEGIAS")
    print("="*60)
    print(f"ARI (CD-HIT vs MMseqs2): {ari_classics:.4f}")
    print(f"ARI (ESM-2 vs CD-HIT):   {ari_esm_vs_cdhit:.4f}")
    print(f"ARI (ESM-2 vs MMseqs2):  {ari_esm_vs_mmseqs:.4f}")
    print("="*60)

    # Guardar resultados para Fase 4 (Validacion Biologica)
    df_meta['cluster_esm'] = labels_esm
    df_meta.to_csv("master_agnostico.csv", index=False)

if __name__ == "__main__":
    benchmark_agnostico()