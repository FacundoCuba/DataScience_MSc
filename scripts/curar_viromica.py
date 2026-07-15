#!/usr/bin/env python3

from pymongo import MongoClient
import time

def curar_datos():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    raw_col = db["genes_virales"]
    clean_col = db["genes_curados"]
    
    # Definimos umbrales
    MIN_LEN = 30
    MAX_LEN = 1022

    print(f"[{time.strftime('%H:%M:%S')}] Iniciando curación de datos...")
    
    # 1. Limpiamos la colección de destino si ya existe
    clean_col.drop()

    # 2. Pipeline de filtrado y transferencia
    pipeline = [
        {
            "$addFields": {
                "largo": {"$strLenCP": "$aa_sequence"}
            }
        },
        {
            "$match": {
                "largo": {"$gte": MIN_LEN, "$lte": MAX_LEN}
            }
        },
        {
            "$out": "genes_curados" # Crea la nueva colección con los resultados
        }
    ]

    raw_col.aggregate(pipeline)
    
    # 3. Verificación
    total_original = raw_col.count_documents({})
    total_curado = clean_col.count_documents({})
    descartados = total_original - total_curado

    print(f"[{time.strftime('%H:%M:%S')}] Curación finalizada.")
    print(f"--- Resumen ---")
    print(f"Registros originales: {total_original}")
    print(f"Registros curados:    {total_curado}")
    print(f"Registros eliminados: {descartados} ({(descartados/total_original)*100:.2f}%)")

    # Creamos índices en la nueva colección para la fase de clustering
    print("Creando índices en 'genes_curados'...")
    clean_col.create_index("protein_id")

if __name__ == "__main__":
    curar_datos()