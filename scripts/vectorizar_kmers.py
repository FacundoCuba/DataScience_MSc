import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import joblib
import time
import os

def generar_kmers_optimizados():
    # Configuración de conexión
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    col = db["genes_curados"]

    total_registros = col.count_documents({})
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando proceso para {total_registros} secuencias.")

    # 1. Definimos un generador para las secuencias
    # Esto evita cargar los 686k strings en la RAM simultáneamente
    def secuencia_generator():
        cursor = col.find({}, {"aa_sequence": 1, "_id": 0})
        for doc in cursor:
            yield doc["aa_sequence"]

    # 2. Configuración del Vectorizador
    # analyzer='char' y ngram_range=(3,3) genera trímeros de aminoácidos
    # dtype=np.uint16 es suficiente para contar repeticiones en proteínas y ahorra RAM
    vectorizer = CountVectorizer(
        analyzer='char', 
        ngram_range=(3, 3), 
        dtype=np.uint16
    )

    print(f"[{time.strftime('%H:%M:%S')}] Ajustando y transformando matriz (3-mers)...")
    try:
        # Fit & Transform usando el generador
        X = vectorizer.fit_transform(secuencia_generator())
        
        print(f"[{time.strftime('%H:%M:%S')}] Matriz generada con éxito.")
        print(f"Dimensiones finales: {X.shape}") # Debería ser (686911, ~8000)

        # 3. Guardado eficiente
        print(f"[{time.strftime('%H:%M:%S')}] Guardando archivos...")
        
        # Guardamos la matriz en formato NPZ (comprimido para matrices dispersas)
        sparse.save_npz("matriz_kmers_3mers.npz", X)
        
        # Extraemos y guardamos los IDs por separado para mantener el orden
        print(f"[{time.strftime('%H:%M:%S')}] Extrayendo IDs para indexación...")
        ids = [doc["protein_id"] for doc in col.find({}, {"protein_id": 1, "_id": 0})]
        joblib.dump(ids, "ids_kmers.pkl")
        
        # Guardamos el vocabulario del vectorizador (opcional, útil para saber qué trímero es cada columna)
        joblib.dump(vectorizer.get_feature_names_out(), "vocabulario_3mers.pkl")

        print(f"[{time.strftime('%H:%M:%S')}] ¡Proceso completado!")
        print(f"Archivos generados: matriz_kmers_3mers.npz, ids_kmers.pkl")

    except Exception as e:
        print(f"Error durante la vectorización: {e}")

if __name__ == "__main__":
    generar_kmers_optimizados()