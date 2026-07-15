#!/usr/bin/env python3

from pymongo import MongoClient

def exportar():
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