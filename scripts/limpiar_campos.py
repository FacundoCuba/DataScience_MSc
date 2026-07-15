from pymongo import MongoClient

def limpiar_campos_duplicados():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    col = db["genes_curados"]

    print("Unificando campos de clusters mmseqs...")
    
    # 1. Copiamos los valores de mmseqs_cluster_90 a cluster_mmseqs_90 si este último no existe
    # 2. Eliminamos mmseqs_cluster_90
    result = col.update_many(
        {"mmseqs_cluster_90": {"$exists": True}},
        [
            {"$set": {"cluster_mmseqs_90": "$mmseqs_cluster_90"}},
            {"$unset": "mmseqs_cluster_90"}
        ]
    )

    print(f"Registros actualizados: {result.modified_count}")

if __name__ == "__main__":
    limpiar_campos_duplicados()