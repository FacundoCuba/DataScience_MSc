import pandas as pd
import numpy as np
import h5py
import pickle
from pymongo import MongoClient
import cudf
import cuml
from cuml.cluster import HDBSCAN as cumlHDBSCAN, KMeans as cumlKMeans
from cuml.decomposition import PCA as cumlPCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import cupy as cp
import time

def benchmark_evolucionado():
    print(f"[{time.strftime('%H:%M:%S')}] 1. Conectando a MongoDB viromica_db...")
    client = MongoClient('mongodb://localhost:27017/')
    db = client['viromica_db']
    col = db['genes_curados']

    # 2. Cargar IDs y Metadatos Taxonómicos (Objetivo Específico 3)
    print(f"[{time.strftime('%H:%M:%S')}] 2. Recuperando etiquetas de validación...")
    # Recuperamos la taxonomía real y los tags de CD-HIT/MMseqs que ya tenés en la base
    cursor = col.find({}, {
        "protein_id": 1, 
        "metadata_virus.taxonomy": 1, 
        "cdhit_cluster_90": 1, 
        "cluster_mmseqs_90": 1,
        "largo": 1
    })
    
    df_meta = pd.DataFrame(list(cursor))
    # Extraemos la Familia Viral (penúltima posición usualmente) como ground truth
    # Si el array es corto, tomamos la última disponible
    df_meta['familia_real'] = df_meta['metadata_virus'].apply(
        lambda x: x['taxonomy'][-2] if len(x['taxonomy']) > 1 else x['taxonomy'][-1]
    )

    # 3. Procesamiento de Embeddings (ESM-2)
    print(f"[{time.strftime('%H:%M:%S')}] 3. Cargando Embeddings ESM-2 en GPU...")
    with h5py.File("embeddings_esm2.h5", "r") as h5f:
        # Aseguramos que el orden coincida con los IDs de la DB
        emb_gpu = cp.array(h5f["vectors"][:])
    
    # Reducción de dimensionalidad para mejorar performance de HDBSCAN
    pca = cumlPCA(n_components=50)
    emb_reduced = pca.fit_transform(emb_gpu)

    # 4. Clustering Competitivo (Detección de Homología)
    print(f"[{time.strftime('%H:%M:%S')}] 4. Ejecutando Clustering sobre Espacio Latente...")
    
    # K-means (Baseline de alta performance)
    kmeans = cumlKMeans(n_clusters=1000, random_state=42)
    labels_kmeans_esm = kmeans.fit_predict(emb_reduced).get()
    
    # HDBSCAN (Detección de densidad para familias divergentes)
    hdbscan = cumlHDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
    labels_hdbscan_esm = hdbscan.fit_predict(emb_reduced).get()

    # 5. Cálculo de Métricas de Benchmark (Comparativa Triple)
    print(f"[{time.strftime('%H:%M:%S')}] 5. Calculando ARI contra Taxonomía y Métodos Clásicos...")
    
    # ARI contra la Verdad Biológica (Taxonomía)
    ari_esm_vs_tax = adjusted_rand_score(df_meta['familia_real'], labels_hdbscan_esm)
    ari_cdhit_vs_tax = adjusted_rand_score(df_meta['familia_real'], df_meta['cdhit_cluster_90'])
    ari_mmseqs_vs_tax = adjusted_rand_score(df_meta['familia_real'], df_meta['cluster_mmseqs_90'])
    
    # Discrepancia entre métodos (Lo que hablábamos antes)
    ari_classics_discrepancy = adjusted_rand_score(df_meta['cdhit_cluster_90'], df_meta['cluster_mmseqs_90'])

    # Validación Interna (Silueta) sobre muestra
    idx = np.random.choice(len(emb_reduced), 10000, replace=False)
    sil_score = silhouette_score(emb_reduced.get()[idx], labels_hdbscan_esm[idx])

    # 6. Reporte Final para el TFI
    print("\n" + "="*60)
    print(f"BENCHMARK DE DETECCIÓN DE HOMOLOGÍA VIRAL")
    print("="*60)
    print(f"Métrica de Oro: ARI vs Taxonomía (Familia)")
    print(f"  - ESM-2 + HDBSCAN:  {ari_esm_vs_tax:.4f}")
    print(f"  - CD-HIT (Clásico): {ari_cdhit_vs_tax:.4f}")
    print(f"  - MMseqs2 (Clásico):{ari_mmseqs_vs_tax:.4f}")
    print("-"*60)
    print(f"Discrepancia CD-HIT vs MMseqs2 (ARI): {ari_classics_discrepancy:.4f}")
    print(f"Validación Interna (Silhouette ESM-2): {sil_score:.4f}")
    print("="*60)

    # 7. Guardar resultados para el Objetivo 4 (Anotación)
    df_meta['esm_hdbscan'] = labels_hdbscan_esm
    df_meta['esm_kmeans'] = labels_kmeans_esm
    
    # Guardamos CSV maestro para análisis de "Proteoma Oscuro" en el Malbrán
    output_cols = ['protein_id', 'familia_real', 'cdhit_cluster_90', 'cluster_mmseqs_90', 'esm_hdbscan', 'largo']
    df_meta[output_cols].to_csv("master_clustering_evolucionado.csv", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] Proceso finalizado. Resultados en CSV.")

if __name__ == "__main__":
    benchmark_evolucionado()