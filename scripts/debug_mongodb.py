from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

print("="*60)
print("1. BASES DE DATOS DISPONIBLES EN MONGODB:")
dbs = client.list_database_names()
print(dbs)
print("="*60)

# Buscamos la base de datos que no sea de sistema ('admin', 'config', 'local')
user_dbs = [db_name for db_name in dbs if db_name not in ['admin', 'config', 'local']]

if not user_dbs:
    print("⚠️ No se encontraron bases de datos de usuario. Revisa el puerto o la instancia.")
else:
    target_db_name = user_dbs[0]
    print(f"2. INSPECCIONANDO BASE DE DATOS: '{target_db_name}'")
    db = client[target_db_name]
    print("Colecciones en esta BD:", db.list_collection_names())
    print("="*60)

    # Revisar genes_curados
    if "genes_curados" in db.list_collection_names():
        doc_gene = db["genes_curados"].find_one()
        print(f"✓ Documento de ejemplo en 'genes_curados':\n{doc_gene}\n")
    else:
        print("❌ La colección 'genes_curados' no está en esta BD.")

    # Revisar vec_prott5
    if "vec_prott5" in db.list_collection_names():
        doc_vec = db["vec_prott5"].find_one()
        print(f"✓ Claves en 'vec_prott5': {list(doc_vec.keys()) if doc_vec else 'Vacía'}")
    else:
        print("❌ La colección 'vec_prott5' no está en esta BD.")