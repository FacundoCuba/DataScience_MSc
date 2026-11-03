#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Inspección Específica por Clúster para Proteínas Virales.
Permite consultar un clúster particular a partir de la partición Leiden ya calculada,
exportando un reporte completo en texto plano con el total de las secuencias.
"""

import os
import argparse
from typing import List, Dict, Tuple
import time
import numpy as np
import pandas as pd
from pymongo import MongoClient


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
    
    protein_to_product = load_annotations_from_genes_curados(db)

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
        raise ValueError(f"No se pudieron extraer vectores de '{vector_col_name}'.")

    X = np.array(vectors, dtype=np.float32)
    y_product = np.array(products)

    tf = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Base vectorial procesada correctamente: {X.shape} (Tiempo total de carga: {tf:.2f}s)")
    return X, y_product, protein_ids


def analizar_cluster_especifico(
    cluster_id_objetivo: int,
    X: np.ndarray,
    labels: np.ndarray,
    y_product: np.ndarray,
    protein_ids: List[str],
    output_report: bool = True
) -> pd.DataFrame:
    """
    Realiza un análisis profundo para un clúster dado, evaluando el total de sus secuencias,
    y genera un reporte en texto plano con el detalle completo.
    """
    print(f"\n============================================================")
    print(f"🔍 ANÁLISIS ESPECÍFICO DEL CLÚSTER ID: {cluster_id_objetivo}")
    print(f"============================================================")
    
    idx_cpu = np.where(labels == cluster_id_objetivo)[0]
    size = len(idx_cpu)
    
    if size == 0:
        print(f"⚠️ El clúster {cluster_id_objetivo} no contiene elementos o no existe.")
        return pd.DataFrame()
    
    cluster_pids = [protein_ids[i] for i in idx_cpu]
    cluster_products = y_product[idx_cpu]
    cluster_emb = X[idx_cpu]
    
    centroid = np.mean(cluster_emb, axis=0)
    dists = np.linalg.norm(cluster_emb - centroid, axis=1)
    
    df_miembros = pd.DataFrame({
        "protein_id": cluster_pids,
        "product": cluster_products,
        "dist_al_centroide": dists
    }).sort_values(by="dist_al_centroide")
    
    print(f"• Tamaño total del clúster: {size:,} proteínas")
    print(f"• Distancia media al centroide: {np.mean(dists):.4f}")
    print(f"• Distancia máxima (Radio): {np.max(dists):.4f}")
    
    print(f"\n📊 Distribución funcional (Top productos):")
    counts = pd.Series(cluster_products).value_counts()
    for prod, cnt in counts.head(10).items():
        pct = (cnt / size) * 100
        print(f"  - {prod}: {cnt} ({pct:.2f}%)")
        
    mask_hipoteticas = df_miembros["product"].str.lower().str.contains("hypothetical|uncharacterized|unknown", na=False)
    n_hipoteticas = mask_hipoteticas.sum()
    print(f"\n🧬 Proteínas hipotéticas / no caracterizadas en el clúster: {n_hipoteticas} ({(n_hipoteticas/size)*100:.2f}%)")
    
    # Generar reporte en texto plano con el total de secuencias
    if output_report:
        report_filename = f"reporte_cluster_{cluster_id_objetivo}.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(f"REPORTE DE INSPECCION - CLUSTER ID: {cluster_id_objetivo}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Tamaño total del clúster: {size:,} proteínas\n")
            f.write(f"Distancia media al centroide: {np.mean(dists):.4f}\n")
            f.write(f"Radio máximo al centroide: {np.max(dists):.4f}\n")
            f.write(f"Proteínas hipotéticas / no caracterizadas: {n_hipoteticas} ({(n_hipoteticas/size)*100:.2f}%)\n\n")
            
            f.write("DISTRIBUCION FUNCIONAL COMPLETA:\n")
            f.write("-" * 50 + "\n")
            for prod, cnt in counts.items():
                pct = (cnt / size) * 100
                f.write(f"{prod}: {cnt} ({pct:.2f}%)\n")
                
            f.write("\n" + "=" * 70 + "\n")
            f.write("DETALLE COMPLETO DE SECUENCIAS DEL CLUSTER (Ordenadas por cercanía al centroide):\n")
            f.write("=" * 70 + "\n")
            f.write(df_miembros.to_string(index=False))
            
        print(f"\n💾 Reporte detallado guardado exitosamente en: '{report_filename}'")
        
    return df_miembros


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspección detallada de un clúster específico por ID.")
    parser.add_argument("cluster_id", type=int, help="ID del clúster que deseas inspeccionar.")
    args = parser.parse_args()
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    
    partition_file = "labels_leiden_prott5.npy"
    
    if not os.path.exists(partition_file):
        raise FileNotFoundError(
            f"No se encontró '{partition_file}'. Ejecutá primero el script principal de clustering."
        )
        
    print(f"🔄 Cargando vectores y mapeos desde MongoDB...")
    X, y_product, protein_ids = load_vectors_and_join(db, vector_col_name="vec_prott5")
    
    print(f"🔄 Cargando etiquetas de Leiden desde disco...")
    labels = np.load(partition_file)
    
    # Ejecutar inspección con el parámetro recibido por consola
    df_detalle = analizar_cluster_especifico(
        cluster_id_objetivo=args.cluster_id,
        X=X,
        labels=labels,
        y_product=y_product,
        protein_ids=protein_ids,
        output_report=True
    )