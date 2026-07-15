import torch
from transformers import AutoTokenizer, EsmModel
from pymongo import MongoClient
import numpy as np
import h5py
from tqdm import tqdm
import time

def generar_embeddings_esm2():
    # 1. Configuración de Dispositivo y Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Usando: {device}")

    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval() # Modo inferencia

    # 2. Conexión a MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    col = db["genes_curados"]

    # 3. Preparación de datos
    cursor = col.find({}, {"protein_id": 1, "aa_sequence": 1, "_id": 0})
    total = col.count_documents({})
    
    batch_size = 128  # Ajustá según la VRAM de tu 5070 (puedes probar 256 o 512)
    
    # 4. Creación del archivo HDF5 para guardar embeddings
    with h5py.File("embeddings_esm2.h5", "w") as h5f:
        # Creamos datasets para los vectores y los IDs
        ds_embeddings = h5f.create_dataset("vectors", (total, 320), dtype='float32')
        ds_ids = h5f.create_dataset("protein_ids", (total,), dtype=h5py.string_dtype())

        print(f"[{time.strftime('%H:%M:%S')}] Iniciando vectorización de {total} secuencias...")

        current_idx = 0
        batch_seqs = []
        batch_ids = []

        for doc in tqdm(cursor, total=total):
            batch_seqs.append(doc['aa_sequence'])
            batch_ids.append(doc['protein_id'])

            if len(batch_seqs) == batch_size or current_idx + len(batch_seqs) == total:
                # Tokenización con padding y truncamiento (ESM2 soporta hasta 1024 tokens)
                inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    # Mean Pooling: Promediamos los embeddings de todos los aminoácidos para tener un vector por proteína
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                # Guardar en HDF5
                num_in_batch = len(batch_ids)
                ds_embeddings[current_idx : current_idx + num_in_batch] = embeddings
                ds_ids[current_idx : current_idx + num_in_batch] = [id.encode('utf8') for id in batch_ids]

                current_idx += num_in_batch
                batch_seqs = []
                batch_ids = []

    print(f"\n[{time.strftime('%H:%M:%S')}] ¡Proceso completado!")
    print("Archivo generado: embeddings_esm2.h5")

if __name__ == "__main__":
    generar_embeddings_esm2()