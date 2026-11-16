"""
Script de Validación Cruzada Definitivo: Clústeres Propios vs. VOGDB
Adaptado al formato real de VOGDB (ProteinIDs con prefijo de especie) y reportes en texto plano.
"""

import os
import glob
import gzip
import time
import pandas as pd


def cargar_indice_vogdb(vog_members_path: str) -> dict:
    """
    Carga vog.members.tsv.gz manejando correctamente la columna ProteinIDs (columna index 4).
    Limpia los prefijos de especie (ej: '2772062.YP_010110418.1' -> 'YP_010110418.1').
    """
    print(f"[{time.strftime('%H:%M:%S')}] 🔄 Cargando y parseando membresías de VOGDB desde '{vog_members_path}'...")
    protein_to_vog = {}
    
    total_lineas = 0
    with gzip.open(vog_members_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            total_lineas += 1
            parts = line.strip().split("\t")
            
            # Verificamos que tenga al menos 5 columnas según el formato especificado
            if len(parts) >= 5:
                vog_id = parts[0].strip()
                proteins_str = parts[4].strip() # La columna ProteinIDs es la quinta (índice 4)
                
                for p_item in proteins_str.split(","):
                    p_clean = p_item.strip()
                    if not p_clean:
                        continue
                    
                    # Guardar con el ID completo por si acaso
                    protein_to_vog[p_clean] = vog_id
                    
                    # Si tiene el formato 'species_id.protein_id', guardamos también la versión limpia sin el punto/prefijo
                    if "." in p_clean:
                        parts_id = p_clean.split(".", 1)
                        if len(parts_id) == 2:
                            protein_to_vog[parts_id[1]] = vog_id
                            
    print(f"✅ VOGDB indexado correctamente: {len(protein_to_vog):,} claves de proteínas asociadas (Analizadas {total_lineas:,} líneas).")
    return protein_to_vog


def parsear_reporte_cluster(filepath: str) -> pd.DataFrame:
    """
    Extrae de forma robusta todos los protein_id de la sección de detalle del reporte.
    """
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
                # La primera columna de la línea tabular es el protein_id
                parts = line_stripped.split()
                if parts:
                    pid = parts[0].strip()
                    if pid and pid.lower() != "protein_id":
                        protein_ids.append(pid)
                        
    return pd.DataFrame({"protein_id": list(set(protein_ids))})


def evaluar_todos_los_reportes(vog_mapping_path: str, output_csv: str = "resumen_validacion_vogdb.csv"):
    t0 = time.perf_counter()
    
    vog_dict = cargar_indice_vogdb(vog_mapping_path)
    
    reportes = glob.glob("reporte_cluster_*.txt")
    if not reportes:
        print("⚠️ No se encontraron archivos 'reporte_cluster_*.txt' en el directorio actual.")
        return
        
    print(f"📂 Evaluando {len(reportes)} clústeres frente a VOGDB...")
    resultados_globales = []
    
    matches_totales = 0
    muestras_impresas = 0
    
    for rep in reportes:
        cluster_id_str = rep.split("_")[-1].replace(".txt", "")
        try:
            cluster_id = int(cluster_id_str)
        except ValueError:
            cluster_id = cluster_id_str
            
        df_miembros = parsear_reporte_cluster(rep)
        if df_miembros.empty:
            continue
            
        total_prots = len(df_miembros)
        vog_ids_asignados = []
        
        for pid in df_miembros["protein_id"]:
            pid_clean = str(pid).strip()
            # Búsqueda exacta o quitando versión si fuera necesario
            vog_assigned = vog_dict.get(pid_clean)
            if not vog_assigned and "." in pid_clean:
                pid_base = pid_clean.split(".")[0]
                vog_assigned = vog_dict.get(pid_base, "Desconocido/Novel")
            elif not vog_assigned:
                vog_assigned = "Desconocido/Novel"
                
            if vog_assigned != "Desconocido/Novel":
                matches_totales += 1
                if muestras_impresas < 5:
                    print(f"   [MATCH EXITOSO] Protein ID '{pid_clean}' -> VOG: '{vog_assigned}'")
                    muestras_impresas += 1
                    
            vog_ids_asignados.append(vog_assigned)
            
        df_miembros["vog_id"] = vog_ids_asignados
        
        conteo_vogs = pd.Series(vog_ids_asignados).value_counts()
        vog_dominante = conteo_vogs.index[0] if len(conteo_vogs) > 0 else "N/A"
        conteo_dominante = conteo_vogs.iloc[0] if len(conteo_vogs) > 0 else 0
        pureza_vog = (conteo_dominante / total_prots) if total_prots > 0 else 0.0
        
        resultados_globales.append({
            "cluster_id": cluster_id,
            "tamaño_cluster": total_prots,
            "vog_dominante": vog_dominante,
            "conteo_en_vog_dominante": conteo_dominante,
            "pureza_ortologa_vog": round(pureza_vog, 4),
            "vogs_diferentes_detectados": len(conteo_vogs)
        })
        
    df_resumen = pd.DataFrame(resultados_globales).sort_values(by="tamaño_cluster", ascending=False)
    df_resumen.to_csv(output_csv, index=False)
    
    tf = time.perf_counter() - t0
    print(f"\n✨ Validación cruzada completada en {tf:.2f}s.")
    print(f"🔗 Total de coincidencias (matches) encontradas con VOGDB: {matches_totales:,}")
    print(f"📊 Resumen consolidado guardado en '{output_csv}'. Total clústeres procesados: {len(df_resumen)}")
    print(df_resumen.head(15).to_string(index=False))


if __name__ == "__main__":
    archivo_vog_members = "VOGDB/vog.members.tsv.gz"
    
    if os.path.exists(archivo_vog_members):
        evaluar_todos_los_reportes(archivo_vog_members)
    else:
        print(f"⚠️ No se encontró el archivo '{archivo_vog_members}'. Verificá la ruta de la carpeta VOGDB.")