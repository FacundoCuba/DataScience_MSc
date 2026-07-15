#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import io
import sys
from Bio import Entrez, SeqIO

# --- Configuración ---
Entrez.email = "facundogcuba@gmail.com"
Entrez.api_key = "339fa88606b213d251e47560a05d917f1f08" 

INPUT_FILE = "accession_list_acotada.txt"
OUTPUT_DIR = "database_gb"
BATCH_SIZE = 100 

def obtener_ruta_archivo(accession, base_dir):
    """Crea una ruta jerárquica para evitar colapsar el sistema de archivos."""
    prefix = accession[:2] 
    sub_prefix = accession[3:5] if len(accession) > 5 else "00"
    path = os.path.join(base_dir, prefix, sub_prefix)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"{accession}.gb")

def descargar_y_fragmentar(accessions, batch_size, output_dir):
    total = len(accessions)
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando descarga de {total} registros...")

    for i in range(0, total, batch_size):
        batch_ids = accessions[i:i+batch_size]
        
        # Filtro de reanudación (Resume)
        ids_a_descargar = [acc for acc in batch_ids if not os.path.exists(obtener_ruta_archivo(acc, output_dir))]
        
        if not ids_a_descargar:
            continue 

        print(f"[{time.strftime('%H:%M:%S')}] Lote {i//batch_size + 1}/{total//batch_size + 1} | Descargando {len(ids_a_descargar)} nuevos...")
        
        reintentos = 0
        while reintentos < 3:
            try:
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=ids_a_descargar,
                    rettype="gbwithparts",
                    retmode="text"
                )
                raw_data = handle.read()
                handle.close()
                
                batch_io = io.StringIO(raw_data)
                for record in SeqIO.parse(batch_io, "genbank"):
                    file_path = obtener_ruta_archivo(record.id, output_dir)
                    with open(file_path, "w") as f_out:
                        SeqIO.write(record, f_out, "genbank")
                break 
                
            except Exception as e:
                reintentos += 1
                print(f"  [ERROR] Lote {i//batch_size + 1} (Intento {reintentos}): {e}")
                time.sleep(15 * reintentos) # Espera incremental
        
        time.sleep(0.3)

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: No se encuentra el archivo {INPUT_FILE}")
        sys.exit(1)

    with open(INPUT_FILE, "r") as f:
        ids = sorted(list(set(line.strip() for line in f if line.strip())))
    
    try:
        descargar_y_fragmentar(ids, BATCH_SIZE, OUTPUT_DIR)
    except KeyboardInterrupt:
        print("\n[!] Descarga interrumpida por el usuario. Puedes reanudar luego.")
        sys.exit(0)