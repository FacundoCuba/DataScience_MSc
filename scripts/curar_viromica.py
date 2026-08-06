#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Filtrado de calidad, curación y persistencia de secuencias de aminoácidos.

Este script representa el paso 4 (Curación de Datos y Control de Calidad) de la 
tubería del TFI. Aplica criterios de filtrado biológico y computacional sobre 
la base de datos cruda 'genes_virales' en MongoDB[cite: 2, 8]. Remueve fragmentos proteicos 
excesivamente cortos y secuencias excesivamente largas para asegurar la compatibilidad 
con el límite de tokens de entrada de ESM-2[cite: 6]. Finalmente, consolida los registros 
válidos en una nueva colección indexada[cite: 5].
"""

from pymongo import MongoClient
import time

def curar_datos() -> None:
    """Ejecuta el pipeline de curación de datos utilizando agregaciones nativas en MongoDB.

    Calcula la longitud de cada secuencia de aminoácidos del lado de la base de datos, 
    descarta los registros que se encuentren fuera del rango definido por `MIN_LEN` (30) 
    y `MAX_LEN` (1175)[cite: 5], y escribe los documentos resultantes de manera atómica 
    en una nueva colección denominada 'genes_curados'[cite: 5]. Al finalizar, calcula métricas 
    del descarte e inicializa un índice para acelerar las búsquedas posteriores[cite: 5].
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    raw_col = db["genes_virales"]
    clean_col = db["genes_curados"]
    
    # Definimos umbrales
    MIN_LEN = 30 # Longitud mínima de aminoácidos para considerar una secuencia como válida desde el punto de vista biológico.
    MAX_LEN = 1175 # Longitud máxima de aminoácidos para considerar una secuencia como válida (3 desvios estándar por encima de la media de ESM-2, que es 1022)[cite: 6].

    print(f"[{time.strftime('%H:%M:%S')}] Iniciando curación de datos...")
    
    # 1. Limpiamos la colección de destino si ya existe
    clean_col.drop()

    # 2. Pipeline de filtrado y transferencia
    pipeline = [
        {
            "$match": {
                "aa_length": {"$gte": MIN_LEN, "$lte": MAX_LEN}
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
    
    if total_original > 0:
        porcentaje_descarte = (descartados / total_original) * 100
        print(f"Registros eliminados: {descartados} ({porcentaje_descarte:.2f}%)")
    else:
        print(f"Registros eliminados: {descartados} (0.00%)")

    # Creamos índices en la nueva colección para la fase de clustering
    print("Creando índices en 'genes_curados'...")
    clean_col.create_index("protein_id")

if __name__ == "__main__":
    curar_datos()