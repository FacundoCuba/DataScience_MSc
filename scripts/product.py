# -*- coding: utf-8 -*-
import pandas as pd
from pymongo import MongoClient
import sys

def validar_pureza_funcional():
    print("--- Fase 4: Validacion Biologica y Funcional ---")
    
    # 1. Conexión a la base de datos de investigación
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['viromica_db']
        col = db['genes_curados']
        print("[OK] Conexion a MongoDB exitosa.")
    except Exception as e:
        print(f"[Error] No se pudo conectar a MongoDB: {e}")
        sys.exit(1)
    
    # 2. Cargar los resultados del benchmark agnóstico
    try:
        df_agnostico = pd.read_csv("master_agnostico.csv")
        # Filtrar el ruido (-1) para enfocarnos en las estructuras encontradas
        df_clusters = df_agnostico[df_agnostico['cluster_esm'] != -1].copy()
        print(f"[OK] Master cargado. Analizando {len(df_clusters)} proteinas agrupadas.")
    except FileNotFoundError:
        print("[Error] No se encontro 'master_agnostico.csv'. Corre el benchmark-3.py primero.")
        sys.exit(1)

    # 3. Identificar los clusters mas significativos de la IA
    # Tomamos los 10 mas grandes para tener una muestra representativa
    top_clusters = df_clusters['cluster_esm'].value_counts().head(10).index.tolist()
    
    reporte_final = []

    print("\nAnalizando consistencia funcional por cluster...")
    for c_id in top_clusters:
        # Extraer IDs de proteinas en este cluster especifico
        proteins_in_cluster = df_clusters[df_clusters['cluster_esm'] == c_id]['protein_id'].tolist()
        
        # Consultar metadatos en MongoDB
        cursor = col.find(
            {"protein_id": {"$in": proteins_in_cluster}}, 
            {"product": 1, "protein_id": 1, "cdhit_cluster_90": 1, "metadata_virus.organism": 1}
        )
        
        data_cluster = pd.DataFrame(list(cursor))
        
        # Calcular metricas de pureza
        total_p = len(data_cluster)
        conteo_productos = data_cluster['product'].value_counts()
        prod_dominante = conteo_productos.index[0] if not conteo_productos.empty else "desconocido"
        pureza = (conteo_productos.iloc[0] / total_p) * 100 if not conteo_productos.empty else 0
        
        # Ver cuantos clusters de CD-HIT "unifico" este cluster de ESM-2
        n_cdhit_orig = data_cluster['cdhit_cluster_90'].nunique()

        print(f"\n" + "="*50)
        print(f"CLUSTER ESM-2 ID: {c_id} | Tamano: {total_p} proteinas")
        print(f"Producto dominante: {prod_dominante} ({pureza:.2f}% de pureza)")
        print(f"Metodos clasicos: Este grupo unifica {n_cdhit_orig} clusters de CD-HIT")
        print("-" * 50)
        print("Top 3 productos encontrados:")
        print(conteo_productos.head(3))
        
        reporte_final.append({
            'cluster_id': c_id,
            'size': total_p,
            'main_product': prod_dominante,
            'purity': pureza,
            'unified_cdhit': n_cdhit_orig
        })

    # 4. Exportar reporte de hallazgos para la tesis
    df_reporte = pd.DataFrame(reporte_final)
    df_reporte.to_csv("reporte_pureza_funcional.csv", index=False)
    print(f"\n[FIN] Reporte guardado en 'reporte_pureza_funcional.csv'.")

if __name__ == "__main__":
    validar_pureza_funcional()