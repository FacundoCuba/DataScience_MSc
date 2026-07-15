import pandas as pd
from pymongo import MongoClient, UpdateOne
import time

def cargar_mmseqs_a_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    col = db["genes_curados"]

    archivo_tsv = "cluster_90_cluster.tsv"
    
    print(f"[{time.strftime('%H:%M:%S')}] Leyendo archivo TSV...")
    # El archivo no tiene headers: col 0 es el Representante, col 1 es el Miembro
    df = pd.read_csv(archivo_tsv, sep='\t', names=['representative', 'member'])

    print(f"[{time.strftime('%H:%M:%S')}] Preparando actualizaciones para {len(df)} registros...")
    
    updates = []
    for _, row in df.iterrows():
        # Usamos UpdateOne para actualizar el campo cluster_mmseqs_90 basado en el protein_id (member)
        updates.append(
            UpdateOne(
                {"protein_id": row['member']}, 
                {"$set": {"cluster_mmseqs_90": row['representative']}}
            )
        )

    print(f"[{time.strftime('%H:%M:%S')}] Ejecutando bulk write en MongoDB...")
    # Procesamos de a 50,000 para no saturar la conexión
    for i in range(0, len(updates), 50000):
        batch = updates[i:i+50000]
        col.bulk_write(batch)
        print(f" Procesados {i + len(batch)}...")

    print(f"[{time.strftime('%H:%M:%S')}] ¡Carga finalizada con éxito!")

if __name__ == "__main__":
    cargar_mmseqs_a_mongo()