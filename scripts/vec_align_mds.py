#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vectorización Masiva por Alineamiento e Inferencia de Landmark MDS mediante CUDA.

Calcula representaciones vectoriales densas para el conjunto completo de secuencias en 
'genes_curados' (690,579) mediante el método de Landmark Multidimensional Scaling (LMDS) 
acelerado por hardware en GPU (PyTorch/CUDA).

Flujo de trabajo:
-----------------
1. Selección de un subconjunto de secuencias de referencia (*Landmarks*, $N_L = 5,000$).
2. Cálculo de la matriz de distancias cruzadas Hamming relativas ($N_L \times N_L$) totalmente 
   vectorizada en GPU con alineamiento y padding dinámico.
3. Estimación del subespacio métrico base ($d=50$) sobre los Landmarks usando Escalado 
   Multidimensional Módulo Métrico (MDS con la seudoinversa de Moore-Penrose).
4. Proyección masiva por lotes (*batches*) e inferencia en GPU de las 690,579 secuencias hacia 
   el subespacio $d=50$ utilizando micro-lotes para evitar desbordamientos de memoria VRAM (OOM).
5. Persistencia de los embeddings continuos (`align_mds_vector`) en la colección `vec_align_mds` 
   de MongoDB con indexación en `protein_id`.
"""

import gc
import time
import numpy as np
import torch
from pymongo import MongoClient
from sklearn.manifold import MDS
from tqdm import tqdm

def get_cuda_device() -> torch.device:
    """Detecta la disponibilidad de un entorno de ejecución con aceleración por GPU CUDA.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[{time.strftime('%H:%M:%S')}] Aceleración CUDA activada: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[{time.strftime('%H:%M:%S')}] CUDA no disponible. Usando CPU.")
    return device

def encode_sequences_dynamic(sequences: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Codifica un lote de cadenas de texto de aminoácidos en tensores PyTorch usando ASCII.

    Aplica un relleno (*padding*) dinámico ajustado exactamente a la longitud de la secuencia
    más larga dentro del lote actual, evitando desperdiciar memoria VRAM con dimensiones fijas
    globales.
    """
    n = len(sequences)
    max_len = max(len(s) for s in sequences) if sequences else 1
    
    encoded = np.zeros((n, max_len), dtype=np.uint8)
    for i, s in enumerate(sequences):
        encoded[i, :len(s)] = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        
    tensor_seqs = torch.tensor(encoded, dtype=torch.int16, device=device)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.float32, device=device)
    return tensor_seqs, lengths

def compute_cross_distances_gpu(
    batch_seqs: list[str], 
    landmark_seqs: list[str], 
    device: torch.device,
    micro_batch_size: int = 50
) -> torch.Tensor:
    """Calcula la matriz de distancias Hamming relativas cruzadas entre dos conjuntos de secuencias en GPU.

    Utiliza micro-lotes y difusión (*broadcasting*) en tensores 3D para evaluar desacoples 
    y normalizar por la longitud máxima de par. Procesa los Landmarks en bloques para acotar 
    el consumo máximo de VRAM.
    """
    n_batch = len(batch_seqs)
    n_landmarks = len(landmark_seqs)
    dist_matrix = torch.zeros((n_batch, n_landmarks), dtype=torch.float32, device=device)

    # Procesamos los landmarks en bloques dinámicos para acotar la dimensión 3D
    lm_block_size = 1000
    for lm_start in range(0, n_landmarks, lm_block_size):
        lm_end = min(lm_start + lm_block_size, n_landmarks)
        sub_lm_seqs = landmark_seqs[lm_start:lm_end]
        
        lm_tensor, lm_lens = encode_sequences_dynamic(sub_lm_seqs, device)
        l_seqs = lm_tensor.unsqueeze(0)  # [1, sub_L, len_lm]
        l_lens = lm_lens.unsqueeze(0)    # [1, sub_L]

        for b_start in range(0, n_batch, micro_batch_size):
            b_end = min(b_start + micro_batch_size, n_batch)
            sub_b_seqs = batch_seqs[b_start:b_end]
            
            b_tensor, b_lens = encode_sequences_dynamic(sub_b_seqs, device)
            
            # Recortamos/Alineamos la longitud máxima común entre ambos sub-grupos
            common_max_len = max(b_tensor.shape[1], lm_tensor.shape[1])
            
            if b_tensor.shape[1] < common_max_len:
                pad = torch.zeros((b_tensor.shape[0], common_max_len - b_tensor.shape[1]), dtype=torch.int16, device=device)
                b_tensor_padded = torch.cat([b_tensor, pad], dim=1)
            else:
                b_tensor_padded = b_tensor

            if lm_tensor.shape[1] < common_max_len:
                pad = torch.zeros((lm_tensor.shape[0], common_max_len - lm_tensor.shape[1]), dtype=torch.int16, device=device)
                l_seqs_padded = torch.cat([lm_tensor, pad], dim=1).unsqueeze(0)
            else:
                l_seqs_padded = l_seqs

            b_sub_seqs = b_tensor_padded.unsqueeze(1) # [subB, 1, common_max_len]
            b_sub_lens = b_lens.unsqueeze(1)           # [subB, 1]

            mismatches = (b_sub_seqs != l_seqs_padded).sum(dim=2, dtype=torch.float32)
            max_lengths = torch.maximum(b_sub_lens, l_lens)
            
            dist_matrix[b_start:b_end, lm_start:lm_end] = mismatches / torch.clamp(max_lengths, min=1.0)

    return dist_matrix

def run_landmark_alignment_mds_full(n_components: int = 50, n_landmarks: int = 5000, batch_size: int = 5000) -> None:
    """Ejecuta la canalización completa de entrenamiento e inferencia masiva de Landmark MDS.

    Extrae los Landmarks, calcula la matriz métrica base en GPU, ajusta la proyección MDS 
    con `scikit-learn` y realiza la transformación lineal masiva para los 690,579 genes, 
    guardando el resultado por lotes en MongoDB.
    """
    device = get_cuda_device()
    
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]
    src_col = db["genes_curados"]
    dst_col = db["vec_align_mds"]

    total_docs = src_col.count_documents({})
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando Landmark MDS en CUDA. Total genes: {total_docs:,}")
    dst_col.drop()

    # 1. Extracción de Landmarks
    print(f"[{time.strftime('%H:%M:%S')}] Extrayendo {n_landmarks} secuencias Landmarks...")
    landmark_docs = list(src_col.find({}, {"protein_id": 1, "aa_sequence": 1, "_id": 0}).limit(n_landmarks))
    landmark_seqs = [d["aa_sequence"] for d in landmark_docs]

    # 2. Computar Matriz Landmark x Landmark (L x L)
    print(f"[{time.strftime('%H:%M:%S')}] Computando matriz de distancias Landmark ({n_landmarks}x{n_landmarks}) en GPU...")
    dist_lm = torch.zeros((n_landmarks, n_landmarks), dtype=torch.float32, device=device)
    
    for i in range(0, n_landmarks, 500):
        end_i = min(i + 500, n_landmarks)
        sub_seqs = landmark_seqs[i:end_i]
        dist_lm[i:end_i] = compute_cross_distances_gpu(sub_seqs, landmark_seqs, device, micro_batch_size=50)

    dist_lm.fill_diagonal_(0.0)
    dist_lm_np = dist_lm.cpu().numpy()
    
    del dist_lm
    torch.cuda.empty_cache()

    # 3. Fit de MDS
    print(f"[{time.strftime('%H:%M:%S')}] Ajustando espacio MDS base con Scikit-Learn...")
    mds = MDS(n_components=n_components, metric=True, init='classical_mds', random_state=42, n_jobs=-1)
    X_landmarks = mds.fit_transform(dist_lm_np)

    landmark_proj_matrix = torch.tensor(np.linalg.pinv(X_landmarks), dtype=torch.float32, device=device)

    # 4. Inferencia e inserción en Mongo
    cursor = src_col.find({}, {"protein_id": 1, "aa_sequence": 1, "_id": 0}, no_cursor_timeout=True).batch_size(batch_size)
    
    inserted = 0
    try:
        with tqdm(total=total_docs, desc="Alineamiento Landmark (CUDA)", unit="seq") as pbar:
            batch_docs = []
            for doc in cursor:
                batch_docs.append(doc)

                if len(batch_docs) >= batch_size:
                    _process_and_insert_batch(
                        batch_docs, landmark_seqs, landmark_proj_matrix, 
                        n_components, device, dst_col
                    )
                    inserted += len(batch_docs)
                    pbar.update(len(batch_docs))
                    batch_docs = []
                    torch.cuda.empty_cache()

            if batch_docs:
                _process_and_insert_batch(
                    batch_docs, landmark_seqs, landmark_proj_matrix, 
                    n_components, device, dst_col
                )
                inserted += len(batch_docs)
                pbar.update(len(batch_docs))
    finally:
        cursor.close()

    print(f"[{time.strftime('%H:%M:%S')}] Indexando 'vec_align_mds'...")
    dst_col.create_index("protein_id")
    print(f"[{time.strftime('%H:%M:%S')}] Proceso completado. Insertados: {inserted:,} documentos.")

def _process_and_insert_batch(
    batch_docs: list[dict], 
    landmark_seqs: list[str], 
    proj_matrix: torch.Tensor, 
    n_components: int, 
    device: torch.device, 
    dst_col
) -> None:
    """Proyecta un lote de documentos al subespacio MDS e inserta los vectores en MongoDB.

    Calcula las distancias desde el lote hacia los Landmarks en GPU, aplica el centrado
    de distancias, realiza la multiplicación matricial con la seudoinversa del espacio Landmark
    y persiste la estructura resultante mediante una inserción masiva.
    """
    protein_ids = [d["protein_id"] for d in batch_docs]
    sequences = [d["aa_sequence"] for d in batch_docs]

    with torch.no_grad():
        dist_batch_lm = compute_cross_distances_gpu(sequences, landmark_seqs, device, micro_batch_size=50)
        centered_dist = dist_batch_lm - dist_batch_lm.mean(dim=1, keepdim=True)
        embeddings = torch.matmul(centered_dist, proj_matrix.T).cpu().numpy()

    mongo_batch = [
        {
            "protein_id": p_id,
            "align_mds_vector": vec.tolist(),
            "dimensions": n_components,
            "model_name": "Landmark MDS"
        }
        for p_id, vec in zip(protein_ids, embeddings)
    ]
    dst_col.insert_many(mongo_batch, ordered=False)

if __name__ == "__main__":
    run_landmark_alignment_mds_full(n_components=50, n_landmarks=5000, batch_size=5000)