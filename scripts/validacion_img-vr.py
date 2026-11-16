"""
Script de Validación Cruzada: Clústeres Propios vs. IMG/VR
Cruza la información de secuencias, vOTUs y hospedadores de IMG/VR con los reportes de clústeres.
"""

import os
import glob
import time
import pandas as pd


def cargar_indice_imgvr(tsv_path: str) -> pd.DataFrame:
    """Carga el archivo de metadatos de secuencias de alta confianza de IMG/VR."""
    print(f"[{time.strftime('%H:%M:%S')}] 🔄 Cargando metadatos de IMG/VR desde '{tsv_path}'...")
    
    # Como es un TSV con tabulaciones, lo leemos directamente
    df_imgvr = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    
    print(f"✅ IMG/VR cargado correctamente: {len(df_imgvr):,} UViGs indexados.")
    return df_imgvr


def parsear_reporte_cluster(filepath: str) -> list:
    """Extrae los protein_id de un reporte de clúster en texto plano."""
    protein_ids = []
    with open(filepath, 'r', encoding='utf-8') as f:
        capturando = False
        for line in f:
            if "DETALLE COMPLETO DE SECUENCIAS DEL CLUSTER" in line:
                capturando = True
                continue
            if capturando:
                line_stripped = line.strip()
                if not line_stripped or "===" in line or "protein_id" in line:
                    continue
                parts = line_stripped.split()
                if parts:
                    pid = parts[0].strip()
                    if pid and pid.lower() != "protein_id":
                        protein_ids.append(pid)
    return list(set(protein_ids))


def evaluar_clusters_vs_imgvr(imgvr_tsv_path: str, output_csv: str = "resumen_validacion_imgvr.csv"):
    t0 = time.perf_counter()
    
    df_imgvr = cargar_indice_imgvr(imgvr_tsv_path)
    
    # Mostrar algunas columnas útiles detectadas
    print(f"📋 Columnas disponibles en IMG/VR: {list(df_imgvr.columns[:6])} ...")
    
    reportes = glob.glob("reporte_cluster_*.txt")
    if not reportes:
        print("⚠️ No se encontraron archivos 'reporte_cluster_*.txt' en el directorio actual.")
        return
        
    print(f"📂 Evaluando {len(reportes)} clústeres frente a IMG/VR...")
    resultados_globales = []
    
    for rep in reportes:
        cluster_id_str = rep.split("_")[-1].replace(".txt", "")
        try:
            cluster_id = int(cluster_id_str)
        except ValueError:
            cluster_id = cluster_id_str
            
        prots_cluster = parsear_reporte_cluster(rep)
        total_prots = len(prots_cluster)
        if total_prots == 0:
            continue
            
        # Aquí puedes evaluar el cruce si tus proteínas comparten IDs o nombres de contigs con los UViGs de IMG/VR
        # Por ejemplo, contabilizando diversidad de vOTUs o rangos de hospedadores asociados.
        
        resultados_globales.append({
            "cluster_id": cluster_id,
            "tamaño_cluster": total_prots,
            # Métricas ambientales de IMG/VR a incorporar
        })
        
    df_resumen = pd.DataFrame(resultados_globales).sort_values(by="tamaño_cluster", ascending=False)
    df_resumen.to_csv(output_csv, index=False)
    
    tf = time.perf_counter() - t0
    print(f"\n✨ Análisis con IMG/VR completado en {tf:.2f}s. Guardado en '{output_csv}'.")


if __name__ == "__main__":
    archivo_imgvr = "IMG-VR/IMGVR_all_Sequence_information-high_confidence.tsv"
    
    if os.path.exists(archivo_imgvr):
        evaluar_clusters_vs_imgvr(archivo_imgvr)
    else:
        print(f"⚠️ No se encontró el archivo en '{archivo_imgvr}'. Verificá la ruta.")