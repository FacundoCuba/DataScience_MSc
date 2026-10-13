"""
Módulo de Caracterización e Inspección por Clúster para Proteínas Virales.

Calcula métricas detalladas por cada clúster generado por la combinación k-NN + Leiden
manteniendo estricta consistencia metodológica con el benchmark general.
Acelera el cálculo geométrico e intrínseco utilizando GPU (CUDA) mediante CuPy/cuML.
"""

import os
from typing import Tuple, List, Dict
import time
import numpy as np
import pandas as pd
from scipy.stats import entropy
from pymongo import MongoClient
import igraph as ig
import leidenalg
from tqdm import tqdm

# Aceleración GPU
import cupy as cp
from cuml.neighbors import NearestNeighbors


def load_annotations_from_genes_curados(db) -> Dict[str, str]:
    """Carga la relación protein_id -> product desde 'genes_curados'."""
    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Cargando anotaciones desde 'genes_curados'...")
    src_col = db["genes_curados"]
    
    mapping = {}
    total_docs = 0
    for doc in src_col.find({}, {"_id": 0, "protein_id": 1, "product": 1}):
        total_docs += 1
        p_id = doc.get("protein_id")
        product = doc.get("product")
        if p_id is not None and product is not None:
            mapping[str(p_id).strip()] = str(product)

    tf = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Documentos leídos: {total_docs:,} | Anotaciones mapeadas: {len(mapping):,} ({tf:.2f}s)")
    return mapping


def load_vectors_and_join(
    db, 
    vector_col_name: str = "vec_prott5"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga todos los vectores de 'vec_prott5' y los cruza con 'genes_curados'."""
    t0 = time.perf_counter()
    
    # 1. Mapeo protein_id -> product
    protein_to_product = load_annotations_from_genes_curados(db)

    # 2. Cargar vectores de ProtT5
    print(f"[{time.strftime('%H:%M:%S')}] Extrayendo vectores desde '{vector_col_name}'...")
    col_vec = db[vector_col_name]

    protein_ids: List[str] = []
    vectors: List[np.ndarray] = []
    products: List[str] = []

    total_vec_docs = 0
    for doc in col_vec.find({}, {"_id": 0, "protein_id": 1, "prott5_vector": 1}):
        total_vec_docs += 1
        p_id = doc.get("protein_id")
        vec = doc.get("prott5_vector")

        if p_id is not None and vec is not None:
            vec_arr = np.array(vec, dtype=np.float32)
            if vec_arr.size > 0:
                p_id_str = str(p_id).strip()
                protein_ids.append(p_id_str)
                vectors.append(vec_arr)
                products.append(protein_to_product.get(p_id_str, "Desconocido"))

    print(f"[{time.strftime('%H:%M:%S')}] Registros vectoriales leídos en '{vector_col_name}': {total_vec_docs:,}")

    if not vectors:
        raise ValueError(
            f"No se pudieron extraer vectores de '{vector_col_name}'."
        )

    X = np.array(vectors, dtype=np.float32)
    y_product = np.array(products)

    tf = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Base vectorial procesada correctamente: {X.shape} (Tiempo total de carga: {tf:.2f}s)")
    return X, y_product, protein_ids


def run_knn_leiden_cuda(
    X: np.ndarray, 
    k_neighbors: int = 15, 
    resolution: float = 0.05
) -> Tuple[ig.Graph, np.ndarray]:
    """Construye el grafo k-NN con cuML (Euclidiana) y ejecuta la partición de Leiden (CPM)."""
    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Construyendo grafo k-NN en GPU (k={k_neighbors}, Euclidiana)...")
    
    t_knn = time.perf_counter()
    X_gpu = cp.asarray(X, dtype=cp.float32)
    
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
    nn.fit(X_gpu)
    distances_gpu, indices_gpu = nn.kneighbors(X_gpu)
    
    distances = cp.asnumpy(distances_gpu)
    indices = cp.asnumpy(indices_gpu)
    print(f"[{time.strftime('%H:%M:%S')}]  └─ k-NN finalizado en {time.perf_counter() - t_knn:.2f}s")
    
    n_samples = X.shape[0]

    t_graph = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Generando aristas del grafo...")
    sources = np.repeat(np.arange(n_samples), k_neighbors)
    targets = indices.reshape(-1)
    weights = 1.0 / (1.0 + distances.reshape(-1))

    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]
    weights = weights[mask]

    print(f"[{time.strftime('%H:%M:%S')}] Construyendo objeto igraph...")
    g = ig.Graph(n=n_samples, edges=list(zip(sources, targets)), directed=False)
    g.es["weight"] = weights
    g.simplify(combine_edges=max)
    print(f"[{time.strftime('%H:%M:%S')}]  └─ Construcción de grafo en {time.perf_counter() - t_graph:.2f}s")

    t_leiden = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Ejecutando algoritmo de Leiden (CPMVertexPartition, resolution={resolution})...")
    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )

    labels = np.array(partition.membership)
    tf = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}]  └─ Leiden completado en {time.perf_counter() - t_leiden:.2f}s")
    print(f"[{time.strftime('%H:%M:%S')}] Clustering completado. Clústeres detectados: {len(np.unique(labels))} (Tiempo etapa: {tf:.2f}s)")
    return g, labels


def calcular_metricas_por_cluster_cuda_checkpoint(
    X: np.ndarray, 
    labels: np.ndarray, 
    y_product: np.ndarray,
    g: ig.Graph,
    checkpoint_file: str = "checkpoint_metricas_prott5.csv",
    batch_save_size: int = 100
) -> pd.DataFrame:
    """Calcula métricas intrínsecas (GPU) y biológicas por cada clúster con guardado incremental."""
    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Calculando métricas intrínsecas y biológicas en GPU...")
    
    unique_clusters = np.unique(labels)
    total_clusters = len(unique_clusters)
    
    # 1. Comprobar si existe un checkpoint previo para reanudar
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        try:
            df_existing = pd.read_csv(checkpoint_file)
            if "cluster_id" in df_existing.columns:
                processed_ids = set(df_existing["cluster_id"].unique())
                print(f"[{time.strftime('%H:%M:%S')}] 🔄 Checkpoint detectado. Reanudando desde el clúster {len(processed_ids):,} / {total_clusters:,}.")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Error al leer el checkpoint existente ({e}). Se iniciará desde cero.")

    # Filtrar los clústeres pendientes
    pending_clusters = [cid for cid in unique_clusters if cid not in processed_ids]
    
    if not pending_clusters:
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Todos los clústeres ya fueron procesados previamente.")
        return pd.read_csv(checkpoint_file).sort_values(by="tamaño", ascending=False).reset_index(drop=True)

    print(f"[{time.strftime('%H:%M:%S')}] Clústeres pendientes a procesar: {len(pending_clusters):,}")

    X_gpu = cp.asarray(X, dtype=cp.float32)
    buffer_metrics = []

    pbar = tqdm(pending_clusters, desc="Procesando clústeres", unit="clúster", leave=True)
    
    try:
        for i, cid in enumerate(pbar):
            idx_cpu = np.where(labels == cid)[0]
            size = len(idx_cpu)
            idx_gpu = cp.asarray(idx_cpu)
            
            # Subconjunto del clúster en GPU
            cluster_emb_gpu = X_gpu[idx_gpu]
            
            # Centroide euclidiano en GPU
            centroid_gpu = cp.mean(cluster_emb_gpu, axis=0, keepdims=True)
            
            # Distancia euclidiana al centroide
            dists_centroid_gpu = cp.linalg.norm(cluster_emb_gpu - centroid_gpu, axis=1)
            
            mean_dist = float(cp.mean(dists_centroid_gpu))
            max_dist = float(cp.max(dists_centroid_gpu))
            var_intra = float(cp.var(cluster_emb_gpu, axis=0).sum())
            
            # Conductancia en el grafo (CPU)
            if size > 1 and g is not None:
                subgraph_edges = g.induced_subgraph(idx_cpu.tolist()).ecount()
                boundary_edges = sum(1 for e in g.es if (e.tuple[0] in idx_cpu) != (e.tuple[1] in idx_cpu))
                total_volume = (2 * subgraph_edges) + boundary_edges
                conductance = boundary_edges / total_volume if total_volume > 0 else 0.0
            else:
                conductance = 0.0

            # Métricas de Anotación (`product`)
            cluster_products = y_product[idx_cpu]
            counts = pd.Series(cluster_products).value_counts()
            top_product = counts.index[0] if len(counts) > 0 else "N/A"
            purity = float(counts.iloc[0] / size) if len(counts) > 0 else 0.0
            
            probs = counts / size
            ent = float(entropy(probs, base=2)) if len(probs) > 0 else 0.0

            row = {
                "cluster_id": int(cid),
                "tamaño": size,
                "pureza_anotacion": round(purity, 4),
                "entropia_anotacion": round(ent, 4),
                "producto_dominante": top_product,
                "dist_media_centroide": round(mean_dist, 4),
                "radio_max_euclidiano": round(max_dist, 4),
                "varianza_latente": round(var_intra, 4),
                "conductancia_grafo": round(conductance, 4)
            }
            buffer_metrics.append(row)

            # Escribir en disco periódicamente (Guardado por lotes)
            if (len(buffer_metrics) >= batch_save_size) or (i == len(pending_clusters) - 1):
                df_batch = pd.DataFrame(buffer_metrics)
                write_header = not os.path.exists(checkpoint_file)
                df_batch.to_csv(checkpoint_file, mode="a", index=False, header=write_header)
                buffer_metrics.clear()  # Vaciar buffer

    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] 🛑 Ejecución pausada manualmente por el usuario (Ctrl+C).")
        if buffer_metrics:
            df_batch = pd.DataFrame(buffer_metrics)
            write_header = not os.path.exists(checkpoint_file)
            df_batch.to_csv(checkpoint_file, mode="a", index=False, header=write_header)
            print(f"[{time.strftime('%H:%M:%S')}] 💾 Último lote no guardado salvado en el checkpoint.")
        print(f"[{time.strftime('%H:%M:%S')}] Podés reanudar el script en cualquier momento ejecutando el mismo comando.")
        raise SystemExit(0)

    # Liberar memoria VRAM
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    df_final = pd.read_csv(checkpoint_file)
    tf = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Cálculo completado exitosamente en {tf:.2f}s ({tf/60:.2f} min).")
    return df_final.sort_values(by="tamaño", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    t_global_start = time.perf_counter()
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    
    # Archivos de persistencia en disco para evitar recalcular
    partition_file = "labels_leiden_prott5.npy"
    sources_file = "sources_knn.npy"
    targets_file = "targets_knn.npy"
    weights_file = "weights_knn.npy"
    
    # 1. Carga vectorial
    X, y_product, valid_ids = load_vectors_and_join(db, vector_col_name="vec_prott5")
    n_samples = X.shape[0]
    
    # 2. Reutilización o cálculo del Grafo + Leiden utilizando archivos guardados
    if all(os.path.exists(f) for f in [partition_file, sources_file, targets_file, weights_file]):
        print(f"[{time.strftime('%H:%M:%S')}] 🔄 Cargando partición y aristas precomputadas desde disco...")
        labels = np.load(partition_file)
        sources = np.load(sources_file)
        targets = np.load(targets_file)
        weights = np.load(weights_file)
        
        print(f"[{time.strftime('%H:%M:%S')}] Reconstruyendo objeto igraph desde aristas guardadas...")
        g = ig.Graph(n=n_samples, edges=list(zip(sources, targets)), directed=False)
        g.es["weight"] = weights
        g.simplify(combine_edges=max)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ⚙️ Ejecutando k-NN y Leiden por primera vez...")
        g, labels = run_knn_leiden_cuda(X, k_neighbors=15, resolution=0.05)
        
        # Extraer aristas del grafo generado para guardarlas
        sources = np.array([e.tuple[0] for e in g.es])
        targets = np.array([e.tuple[1] for e in g.es])
        weights = np.array(g.es["weight"])
        
        # Guardar todo en disco
        np.save(partition_file, labels)
        np.save(sources_file, sources)
        np.save(targets_file, targets)
        np.save(weights_file, weights)
        print(f"[{time.strftime('%H:%M:%S')}] 💾 Partición y aristas guardadas en disco exitosamente.")

    # 3. Cálculo de métricas con Checkpoint Activo
    checkpoint_csv = "reporte_metricas_clusters_prott5.csv"
    df_reporte = calcular_metricas_por_cluster_cuda_checkpoint(
        X, labels, y_product, g, 
        checkpoint_file=checkpoint_csv, 
        batch_save_size=100
    )
    
    t_global_total = time.perf_counter() - t_global_start
    print(f"[{time.strftime('%H:%M:%S')}] Proceso finalizado. Reporte consolidado en '{checkpoint_csv}'. Tiempo total: {t_global_total:.2f}s")
    print(df_reporte.head(15))