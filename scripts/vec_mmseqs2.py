#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorización de Secuencias mediante Clustering Multiresolución con MMseqs2 (Linclust).

Optimizado para escalado masivo (690k+ secuencias) mediante agrupamiento
lineal a múltiples umbrales de identidad (0.3 a 1.0).
"""

import os
import time
import subprocess
import tempfile
from pymongo import MongoClient
from tqdm import tqdm


def parse_mmseqs_tsv(tsv_path: str) -> dict[str, int]:
    """Parsea el resultado de tsvdb de MMseqs2 mapeando {protein_id: cluster_id}."""
    cluster_map = {}
    current_cluster_idx = 0
    repr_to_cluster = {}

    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep_id, member_id = parts[0], parts[1]
                if rep_id not in repr_to_cluster:
                    repr_to_cluster[rep_id] = current_cluster_idx
                    current_cluster_idx += 1
                cluster_map[member_id] = repr_to_cluster[rep_id]
    return cluster_map


def run_mmseqs2_vectorization(
    db_name: str = "viromica_db",
    src_collection: str = "genes_curados",
    dst_collection: str = "vec_mmseqs2",
    thresholds: list[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
    print(f"[{time.strftime('%H:%M:%S')}] Total de secuencias a procesar: {total_docs:,}")

    profiles = {p_id: [] for p_id in protein_ids}

    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path = os.path.join(tmp_dir, "dataset.fasta")

        print(f"[{time.strftime('%H:%M:%S')}] Escribiendo FASTA temporal...")
        with open(fasta_path, "w") as f:
            for d in docs:
                f.write(f">{d['protein_id']}\n{d['aa_sequence'].upper()}\n")

        for c in sorted(thresholds):
            out_cluster = os.path.join(tmp_dir, f"cluster_c_{c}")
            out_tsv = os.path.join(tmp_dir, f"cluster_c_{c}.tsv")
            tmp_mmseqs = os.path.join(tmp_dir, f"tmp_{c}")

            # mmseqs easy-linclust escala linealmente en tiempo y memoria O(N)
            cmd_cluster = [
                "mmseqs", "easy-linclust",
                fasta_path, out_cluster, tmp_mmseqs,
                "--min-seq-id", str(c),
                "-c", "0.8",  # Cobertura mínima de alineamiento (80%)
                "--cov-mode", "0"
            ]

            print(f"[{time.strftime('%H:%M:%S')}] Ejecutando MMseqs2 Linclust (min-seq-id={c})...")
            subprocess.run(cmd_cluster, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # El archivo generado por easy-linclust termina en _cluster.tsv
            cluster_tsv_file = f"{out_cluster}_cluster.tsv"
            cluster_map = parse_mmseqs_tsv(cluster_tsv_file)

            for p_id in protein_ids:
                profiles[p_id].append(cluster_map.get(p_id, -1))

    print(f"[{time.strftime('%H:%M:%S')}] Insertando vectores MMseqs2 en MongoDB...")
    mongo_batch = []
    for p_id in tqdm(protein_ids, desc="Insertando MMseqs2"):
        mongo_batch.append({
            "protein_id": p_id,
            "mmseqs2_vector": profiles[p_id],
            "thresholds_eval": sorted(thresholds),
            "model_name": "MMseqs2_MultiThreshold"
        })

        if len(mongo_batch) >= 5000:
            dst_col.insert_many(mongo_batch, ordered=False)
            mongo_batch = []

    if mongo_batch:
        dst_col.insert_many(mongo_batch, ordered=False)

    print(f"[{time.strftime('%H:%M:%S')}] Proceso finalizado. Registros guardados en '{dst_collection}'.")


if __name__ == "__main__":
    run_mmseqs2_vectorization()