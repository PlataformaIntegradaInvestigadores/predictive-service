# preprocess_data.py

import pandas as pd

# --- CONFIGURACIÓN ---
# 1. Reemplaza con el nombre de tu archivo CSV de datos brutos
RAW_DATA_FILE = "publication_data_p.csv"
# 2. Este será el archivo de salida que usará tu backend
OUTPUT_DATA_FILE = "publication_data.csv"

print(f"Cargando datos brutos desde '{RAW_DATA_FILE}'...")

try:
    # Cargar el dataset original
    df = pd.read_csv(RAW_DATA_FILE)

    # --- PREPROCESAMIENTO ---

    # 1. Convertir 'publication_date' a formato de fecha y extraer el año.
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df.dropna(subset=["publication_date"], inplace=True)
    df["year"] = df["publication_date"].dt.year
    df["year"] = df["year"].astype(int)

    print("Fechas procesadas y año extraído.")

    # 2. Verificar que las columnas necesarias existan
    required_cols = ["affiliation_name", "year", "article_id", "author_id"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"La columna requerida '{col}' no se encuentra en el CSV.")

    print("Agrupando y agregando datos por afiliación y año...")

    # 3. Agrupar por afiliación y año.
    #    - Contar el número de publicaciones (publication_count).
    #    - Contar el número de autores únicos (distinct_authors).
    aggregated_df = (
        df.groupby(["affiliation_name", "year"])
        .agg(
            publication_count=(
                "article_id",
                "nunique",
            ),  # CORREGIDO: Usamos 'article_id' para contar publicaciones únicas
            distinct_authors=(
                "author_id",
                "nunique",
            ),  # CORREGIDO: Usamos 'author_id' para contar autores únicos
        )
        .reset_index()
    )

    print("Datos agregados exitosamente.")

    # 4. Guardar el DataFrame procesado en un nuevo archivo CSV.
    aggregated_df.to_csv(OUTPUT_DATA_FILE, index=False)

    print(f"¡Éxito! Archivo preprocesado guardado como '{OUTPUT_DATA_FILE}'.")
    print("\nColumnas resultantes:")
    print(aggregated_df.columns.tolist())
    print("\nEjemplo de los datos generados:")
    print(aggregated_df.head())

except FileNotFoundError:
    print(
        f"ERROR: No se encontró el archivo '{RAW_DATA_FILE}'. "
        f"Asegúrate de que esté en la misma carpeta."
    )
except KeyError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")
