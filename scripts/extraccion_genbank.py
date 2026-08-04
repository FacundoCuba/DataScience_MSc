#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Procesamiento paralelo e ingesta a MongoDB de secuencias codificantes (CDS).

Este script representa el paso 2 (ETL y Normalización) de la tubería del TFI. 
Recorre de manera recursiva el árbol de directorios jerárquico generado en el paso 1, 
extrae las regiones codificantes (CDS) de los archivos GenBank de forma paralela 
(utilizando multiprocessing) y realiza una ingesta masiva (bulk upsert) en MongoDB 
asegurando la integridad de los datos taxonómicos, genómicos y de traducción[cite: 2, 8].
"""

import os
import io
import time
import logging
from Bio import SeqIO
from pymongo import MongoClient, UpdateOne
from concurrent.futures import ProcessPoolExecutor

# --- Configuracion ---
GB_DIR = "/mnt/c/Users/fcuba/Desktop/Msc Data Science - UNAJ/TFI/database_gb"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "viromica_db"
COLLECTION_NAME = "genes_virales"
BATCH_SIZE = 500  
LOG_FILE = "fase2_extraccion_simple.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def procesar_archivo_gb(file_path: str) -> list:
    """Extrae CDS de un archivo GenBank de forma resiliente.

    Lee el archivo en modo binario para prevenir fallos por códecs corruptos,
    parsea las anotaciones taxonómicas, el hospedador, y extrae los fragmentos
    CDS con sus secuencias nucleotídicas, traducciones proteicas y coordenadas de posición.

    Args:
        file_path (str): Ruta completa al archivo GenBank (.gb) a procesar.

    Returns:
        list of dict: Una lista de diccionarios, donde cada elemento representa
            un gen/proteína (CDS) listo para ser indexado en MongoDB. Devuelve
            una lista vacía si ocurre un error irrecuperable de lectura.
    """
    genes_del_archivo = []
    try:
        # Lectura binaria para evitar errores de codec
        with open(file_path, "rb") as b_handle:
            raw_data = b_handle.read().decode("latin-1", errors="replace")
        
        with io.StringIO(raw_data) as handle:
            for record in SeqIO.parse(handle, "genbank"):
                taxon_id = "unknown"
                host = "unknown"
                for feature in record.features:
                    if feature.type == "source":
                        taxon_refs = feature.qualifiers.get("db_xref", [])
                        for ref in taxon_refs:
                            if "taxon" in ref:
                                taxon_id = ref.split(":")[1]
                        host = feature.qualifiers.get("host", ["unknown"])[0]

                organism = record.annotations.get("source", "unknown")
                taxonomy = record.annotations.get("taxonomy", [])

                for feature in record.features:
                    if feature.type == "CDS":
                        try:
                            nt_seq = str(feature.extract(record.seq))
                        except:
                            continue 

                        aa_seq = feature.qualifiers.get("translation", [""])[0]
                        protein_id = feature.qualifiers.get("protein_id", ["no_id"])[0]
                        product = feature.qualifiers.get("product", ["unknown"])[0]

                        gene_doc = {
                            "protein_id": protein_id,
                            "product": product,
                            "nt_sequence": nt_seq,
                            "aa_sequence": aa_seq,
                            "nt_length": len(nt_seq),
                            "aa_length": len(aa_seq),
                            "metadata_virus": {
                                "accession": record.id,
                                "taxon_id": taxon_id,
                                "organism": organism,
                                "host": host,
                                "taxonomy": taxonomy
                            },
                            "location": {
                                "start": int(feature.location.start),
                                "end": int(feature.location.end),
                                "strand": int(feature.location.strand)
                            }
                        }
                        genes_del_archivo.append(gene_doc)
    except Exception as e:
        logging.error(f"Error en {file_path}: {e}")
    
    return genes_del_archivo


def main() -> None:
    """Orquesta la lectura paralela de los archivos y realiza la ingesta masiva en base de datos.

    Escanea de forma recursiva el directorio raíz `GB_DIR` para identificar los archivos GenBank.
    Inicializa un `ProcessPoolExecutor` para procesar múltiples archivos simultáneamente.
    Finalmente, gestiona un búfer de escritura que ejecuta operaciones de tipo Upsert masivas
    (Bulk Operations) contra MongoDB, manteniendo índices optimizados[cite: 2, 8].
    """
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando extraccion paralela...")
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]
    
    # Asegurar índice para que el UPSERT no sea lento
    col.create_index([("protein_id", 1), ("metadata_virus.accession", 1)])
    
    all_files = []
    for root, _, files in os.walk(GB_DIR):
        for f in files:
            if f.lower().endswith((".gb", ".genbank")):
                all_files.append(os.path.join(root, f))
    
    print(f"Archivos a procesar: {len(all_files)}")

    # Paralelizamos el parsing de archivos
    # Usamos la mitad de tus núcleos para no saturar el bus de datos
    with ProcessPoolExecutor(max_workers=12) as executor:
        bulk_ops = []
        contador_total = 0
        
        for i, genes in enumerate(executor.map(procesar_archivo_gb, all_files), 1):
            for g in genes:
                op = UpdateOne(
                    {
                        "protein_id": g["protein_id"], 
                        "metadata_virus.accession": g["metadata_virus"]["accession"]
                    },
                    {"$set": g},
                    upsert=True
                )
                bulk_ops.append(op)
                
                if len(bulk_ops) >= BATCH_SIZE:
                    col.bulk_write(bulk_ops, ordered=False)
                    contador_total += len(bulk_ops)
                    bulk_ops = []

            if i % 100 == 0:
                print(f"Procesados {i}/{len(all_files)} archivos... (Genes insertados/upd: {contador_total})")

        if bulk_ops:
            col.bulk_write(bulk_ops, ordered=False)
            contador_total += len(bulk_ops)

    client.close()
    print(f"[{time.strftime('%H:%M:%S')}] Finalizado. Total: {contador_total}")


if __name__ == "__main__":
    main()