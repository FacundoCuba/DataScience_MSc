#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorización de Secuencias mediante Perfil de Clusters Multiresolución con CD-HIT.

Ejecuta CD-HIT en múltiples umbrales de identidad (0.4 a 1.0) para construir 
un vector de pertenencia categórica/one-hot por secuencia.
"""

import os
import time
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pymongo import MongoClient
from tqdm import tqdm


def parse_cdhit_clstr_fast(clstr_path: str) -> dict[str, int]:
    """Parsea el archivo .clstr reduciendo operaciones de strings en memoria."""
    cluster_map = {}
    current_cluster = -1

    with open(clstr_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            else:
                try:
                    start_idx = line.index(">") + 1
                    end_idx = line.index("...", start_idx)
                    cluster_map[line[start_idx:end_idx]] = current_cluster
                except ValueError:
                    continue
    return cluster_map


def _run_single_cdhit(args: tuple) -> tuple[float, dict[str, int]]:
    """Función worker para ejecutar un umbral de CD-HIT en paralelo."""
    c, fasta_path, tmp_dir, threads_per_job = args
    out_prefix = os.path.join(tmp_dir, f"cdhit_c_{c}")
    clstr_file = f"{out_prefix}.clstr"

    if c >= 0.7:
        n_word = 5
    elif c >= 0.6:
        n_word = 4
    elif c >= 0.5:
        n_word = 3
    else:
        n_word = 2

    cmd = [
        "cd-hit",
        "-i", fasta_path,
        "-o", out_prefix,
        "-c", str(c),
        "-n", str(n_word),
        "-M", "0",
        "-T", str(threads_per_job)
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cluster_map = parse_cdhit_clstr_fast(clstr_file)
    return c, cluster_map


def run_cdhit_vectorization(
    db_name: str = "viromica_db",
    src_collection: str = "genes_curados",
    dst_collection: str = "vec_cdhit",
    thresholds: list[float] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    max_workers: int = 2
) -> None:
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client[db_name]
    src_col = db[src_collection]
    dst_col = db[dst_collection]

    dst_col.create_index("protein_id")

    print(f"[{time.strftime('%H:%M:%S')}] Extrayendo secuencias de MongoDB...")
    docs = list(src_col.find({}, {"protein_id": 1, "aa_sequence": 1, "_id": 0}))
    total_docs = len(docs)

    if total_docs == 0:
        print(f"[{time.strftime('%H:%M:%S')}] No se encontraron secuencias en '{src_collection}'.")
        return

    protein_ids = [d["protein_id"] for d in docs]
    print(f"[{time.strftime('%H:%M:%S')}] Total de secuencias: {total_docs:,}")

    # Calcular hilos por proceso (CPU totales / max_workers)
    total_cpus = os.cpu_count() or 4
    threads_per_job = max(1, total_cpus // max_workers)

    results_by_c = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path = os.path.join(tmp_dir, "dataset.fasta")

        print(f"[{time.strftime('%H:%M:%S')}] Escribiendo FASTA temporal...")
        with open(fasta_path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(f">{d['protein_id']}\n{d['aa_sequence'].upper()}\n")

        tasks = [
            (c, fasta_path, tmp_dir, threads_per_job)
            for c in sorted(thresholds)
        ]

        print(f"[{time.strftime('%H:%M:%S')}] Lanzando CD-HIT en paralelo ({max_workers} procesos, {threads_per_job} hilos c/u)...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_single_cdhit, task) for task in tasks]
            for future in as_completed(futures):
                c_val, cluster_map = future.result()
                results_by_c[c_val] = cluster_map
                print(f"[{time.strftime('%H:%M:%S')}] Umbral c={c_val} completado.")

    print(f"[{time.strftime('%H:%M:%S')}] Ensamblando perfiles multiresolución...")
    sorted_th = sorted(thresholds)
    mongo_batch = []

    for p_id in tqdm(protein_ids, desc="Insertando CD-HIT"):
        vector = [results_by_c[c].get(p_id, -1) for c in sorted_th]
        mongo_batch.append({
            "protein_id": p_id,
            "cdhit_vector": vector,
            "thresholds_eval": sorted_th,
            "model_name": "CDHIT_MultiThreshold"
        })

        if len(mongo_batch) >= 5000:
            dst_col.insert_many(mongo_batch, ordered=False)
            mongo_batch = []

    if mongo_batch:
        dst_col.insert_many(mongo_batch, ordered=False)

    print(f"[{time.strftime('%H:%M:%S')}] Proceso finalizado en '{dst_collection}'.")


if __name__ == "__main__":
    run_cdhit_vectorization(max_workers=4)