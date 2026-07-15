from pymongo import MongoClient
import pprint

def ver_ejemplo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    col = db["genes_curados"]

    # Buscamos un registro que ya tenga el cluster de MMseqs2
    # Si ya corriste el script de CD-HIT o K-mers, también aparecerán aquí
    documento = col.find_one({"cluster_mmseqs_90": {"$exists": True}})

    if documento:
        print("--- ESTRUCTURA ACTUAL DEL REGISTRO EN MONGODB ---")
        pprint.pprint(documento)
    else:
        print("No se encontró ningún registro con clusters aún.")

if __name__ == "__main__":
    ver_ejemplo()