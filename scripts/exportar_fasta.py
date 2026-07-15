#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Exportación de secuencias de aminoácidos curadas a formato estándar FASTA.

Este script representa el paso 6 (Interoperabilidad y Exportación de Datos) de la 
tubería del TFI. Lee los documentos procesados desde la colección 'genes_curados' 
en MongoDB y los convierte al formato FASTA[cite: 5]. Este paso asegura la compatibilidad 
del dataset con herramientas bioinformáticas tradicionales de alineamiento y clustering 
secuencial clásico (como CD-HIT o MMseqs2) en las fases posteriores de validación.
"""

from pymongo import MongoClient


def exportar() -> None:
    """Exporta las proteínas curadas desde MongoDB hacia un archivo plano en formato FASTA.

    Realiza una consulta optimizada sobre la colección 'genes_curados', trayendo 
    exclusivamente mediante proyección el identificador de la proteína y su secuencia[cite: 5]. 
    Escribe cada registro siguiendo la nomenclatura formal de cabecera de FASTA ('>ID_PROTEINA' 
    seguido por la secuencia de caracteres de aminoácidos en la siguiente línea).

    Files generated:
        genes_curados.fasta (text file): Archivo multifasta conteniendo todas las 
            secuencias curadas del dataset listas para su procesamiento bioinformático.

    Raises:
        pymongo.errors.ConnectionFailure: Si se pierde la comunicación con el motor local de MongoDB.
        KeyError: Si algún documento de la colección carece de las propiedades 'protein_id' 
            o 'aa_sequence'.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    col = db["genes_curados"]
    
    print("Exportando genes curados a FASTA...")
    with open("genes_curados.fasta", "w") as f:
        for doc in col.find({}, {"protein_id": 1, "aa_sequence": 1}):
            # Usamos el protein_id como cabecera
            f.write(f">{doc['protein_id']}\n{doc['aa_sequence']}\n")
    print("Listo: genes_curados.fasta")


if __name__ == "__main__":
    exportar()