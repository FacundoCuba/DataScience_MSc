#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Análisis Exploratorio de Datos (EDA) sobre el conjunto de datos curado.

Este script representa el paso 5 (Análisis Exploratorio Post-Curación) de la 
tubería del TFI. Realiza un diagnóstico descriptivo y visual idéntico al del paso 3, 
pero enfocado exclusivamente en la colección 'genes_curados'[cite: 5]. Esto permite 
evaluar el impacto del filtrado de longitudes (rango [30, 1022] aa) sobre la 
composición taxonómica, la distribución de hospedadores y la estructura general 
de los datos biológicos que ingresarán a los modelos de representación (embeddings)[cite: 5].
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient

def run_eda() -> None:
    """Ejecuta el análisis exploratorio sobre los datos curados y genera gráficos de diagnóstico.

    Se conecta a la colección 'genes_curados' en MongoDB y procesa la distribución de 
    longitudes de secuencia, así como el conteo de organismos y hospedadores[cite: 5]. 
    Genera un lienzo con cuatro visualizaciones clave (histograma, boxplot, gráfico de 
    barras y gráfico de torta) para el informe de avance.
    """
    # 1. Conexión
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    col = db["genes_curados"]

    print("--- Iniciando Análisis Exploratorio de Datos ---")

    # A. Distribución de Longitudes (Aminoácidos)
    # Extraemos solo las longitudes para no saturar la RAM
    print("[1/4] Analizando longitudes de secuencias...")
    pipeline_len = [
        {"$project": {"length": {"$strLenCP": "$aa_sequence"}}}
    ]
    lengths = [doc['length'] for doc in col.aggregate(pipeline_len)]
    df_len = pd.DataFrame(lengths, columns=['aa_length'])

    # B. Top 15 Organismos (Virus)
    print("[2/4] Contando taxones más frecuentes...")
    pipeline_taxa = [
        {"$group": {"_id": "$metadata_virus.organism", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]
    taxa_data = list(col.aggregate(pipeline_taxa))
    df_taxa = pd.DataFrame(taxa_data).rename(columns={'_id': 'Organism', 'count': 'Frequency'})

    # C. Análisis de Hosts (Huéspedes)
    print("[3/4] Analizando distribución de hosts...")
    pipeline_host = [
        {"$group": {"_id": "$metadata_virus.host", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    host_data = list(col.aggregate(pipeline_host))
    df_host = pd.DataFrame(host_data).rename(columns={'_id': 'Host', 'count': 'Frequency'})

    # --- Visualización ---
    print("[4/4] Generando gráficos...")
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Gráfico 1: Histograma de longitudes
    sns.histplot(df_len['aa_length'], bins=100, kde=True, ax=axes[0, 0], color='teal')
    axes[0, 0].set_title('Distribución de Longitudes (aa)')
    axes[0, 0].set_xlim(0, df_len['aa_length'].quantile(0.95)) # Recortamos outliers para ver mejor
    axes[0, 0].set_xlabel('Cantidad de Aminoácidos')

    # Gráfico 2: Top Organismos
    sns.barplot(data=df_taxa, y='Organism', x='Frequency', ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title('Top 15 Organismos en el Data Lake')

    # Gráfico 3: Distribución de Hosts
    axes[1, 0].pie(df_host['Frequency'], labels=df_host['Host'], autopct='%1.1f%%', startangle=140)
    axes[1, 0].set_title('Distribución de Huéspedes (Top 10)')

    # Gráfico 4: Boxplot de longitudes
    sns.boxplot(x=df_len['aa_length'], ax=axes[1, 1], color='coral')
    axes[1, 1].set_title('Boxplot de Longitudes (Detección de Outliers)')
    axes[1, 1].set_xlim(0, df_len['aa_length'].quantile(0.95))

    plt.tight_layout()
    plt.savefig('eda_viromica_curados_results.png')
    print("\n--- EDA Finalizado ---")
    print("Resultados guardados en 'eda_viromica_curados_results.png'")
    
    # Estadísticas rápidas por consola
    print(f"\nEstadísticas de Longitud:")
    print(df_len.describe())

def generar_reporte_texto() -> None:
    """Genera un informe detallado de estadísticas descriptivas en formato de texto plano.

    Extrae las métricas globales de longitud, listado de organismos representados 
    y hospedadores sobre los datos filtrados en la colección 'genes_curados'[cite: 5]. 
    Facilita el contraste empírico de volumen de pérdida frente al reporte crudo.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["viromica_db"]
    col = db["genes_curados"]

    print("--- Generando Reporte de Texto Plano ---")

    # 1. Estadísticas de Longitud
    pipeline_len = [{"$project": {"largo": {"$strLenCP": "$aa_sequence"}}}]
    lengths = [doc['largo'] for doc in col.aggregate(pipeline_len)]
    df_len = pd.DataFrame(lengths, columns=['aa_length'])
    stats = df_len.describe()

    # 2. Top Organismos
    pipeline_taxa = [
        {"$group": {"_id": "$metadata_virus.organism", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]
    df_taxa = pd.DataFrame(list(col.aggregate(pipeline_taxa)))

    # 3. Top Hosts
    pipeline_host = [
        {"$group": {"_id": "$metadata_virus.host", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    df_host = pd.DataFrame(list(col.aggregate(pipeline_host)))

    # --- Escritura del Archivo ---
    with open("reporte_eda_viromica_curados.txt", "w", encoding="utf-8") as f:
        f.write("==========================================\n")
        f.write("   REPORTE EXPLORATORIO: VIROMICA_DB (CURADOS)\n")
        f.write(f"   Total de registros: {len(df_len)}\n")
        f.write("==========================================\n\n")

        f.write("1. ESTADISTICAS DE LONGITUD (Aminoacidos)\n")
        f.write("------------------------------------------\n")
        f.write(stats.to_string())
        f.write("\n\n")

        f.write("2. TOP 15 ORGANISMOS (Frecuencia de Genes)\n")
        f.write("------------------------------------------\n")
        for _, row in df_taxa.iterrows():
            f.write(f"{str(row['_id']):<50} | {row['count']}\n")
        f.write("\n")

        f.write("3. DISTRIBUCION DE HOSTS (Huespedes)\n")
        f.write("------------------------------------------\n")
        for _, row in df_host.iterrows():
            f.write(f"{str(row['_id']):<50} | {row['count']}\n")

    print("Reporte guardado exitosamente en 'reporte_eda_viromica_curados.txt'")

if __name__ == "__main__":
    run_eda()
    generar_reporte_texto()