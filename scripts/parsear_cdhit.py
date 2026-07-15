from pymongo import MongoClient
import re

def cargar_cdhit_a_mongo(file_path):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    col = db["genes_curados"]

    print("Parseando archivo .clstr de CD-HIT...")
    
    current_cluster = None
    updates = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>Cluster'):
                current_cluster = line.strip().split()[-1]
            else:
                # Extraer el ID de la proteína entre > y ...
                match = re.search(r'>(.*?)\.\.\.', line)
                if match:
                    protein_id = match.group(1)
                    updates.append((protein_id, current_cluster))

    print(f"Actualizando {len(updates)} registros en MongoDB...")
    # Aquí podés usar bulk_write como hicimos antes para que sea rápido
    for p_id, cluster_id in updates:
        col.update_one({"protein_id": p_id}, {"$set": {"cdhit_cluster_90": cluster_id}})

    print("¡Finalizado!")

if __name__ == "__main__":
    cargar_cdhit_a_mongo("cdhit_result_90.clstr")