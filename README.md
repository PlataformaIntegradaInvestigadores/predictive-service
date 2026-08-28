# Predictive Model Backend

Este proyecto es un backend de FastAPI diseñado para predecir el número de publicaciones científicas de diversas afiliaciones académicas y de investigación. Utiliza un modelo de Machine Learning (LightGBM Regressor) pre-entrenado para ofrecer proyecciones, comparaciones y rankings.

## Tabla de Contenidos

- [Predictive Model Backend](#predictive-model-backend)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [Propósito del Proyecto](#propósito-del-proyecto)
  - [Funcionamiento](#funcionamiento)
  - [Instalación](#instalación)
    - [Requisitos Previos](#requisitos-previos)
    - [Configuración del Entorno](#configuración-del-entorno)
  - [Uso](#uso)
    - [Ejecución en Local](#ejecución-en-local)
    - [Ejecución con Docker](#ejecución-con-docker)
  - [Endpoints de la API](#endpoints-de-la-api)
  - [Estructura del Repositorio](#estructura-del-repositorio)

## Propósito del Proyecto

El objetivo principal de este backend es proporcionar una herramienta analítica que permita a los usuarios:

- Obtener una lista de todas las afiliaciones de investigación disponibles en el conjunto de datos.
- Visualizar la tendencia histórica de publicaciones de una afiliación específica.
- Predecir el número futuro de publicaciones para una o varias afiliaciones.
- Realizar análisis "What If" para estimar cómo cambios en el número de autores podrían impactar las publicaciones futuras.
- Comparar las proyecciones de crecimiento entre múltiples afiliaciones.
- Obtener un ranking de las afiliaciones con mayor crecimiento proyectado.

## Funcionamiento

El núcleo del backend es el `PredictionService`, que se encarga de:

1.  **Cargar los Activos**: Al iniciar, el servicio carga el modelo de Machine Learning (`publication_model.pkl`), el codificador de afiliaciones (`affiliation_encoder.pkl`) y el conjunto de datos históricos (`publication_data.csv`).
2.  **Procesar Solicitudes**: La API, construida con FastAPI, recibe las solicitudes a través de diferentes endpoints definidos en `analytics.py`.
3.  **Realizar Predicciones**: Para las proyecciones a futuro, el servicio utiliza el modelo pre-entrenado para predecir el número de publicaciones para los años solicitados, basándose en datos históricos como el número de publicaciones y autores del último año registrado.
4.  **Devolver Resultados**: Los resultados se formatean según los esquemas definidos en `schemas.py` y se devuelven como respuestas JSON.

## Instalación

Sigue estos pasos para configurar el proyecto en tu entorno local.

### Requisitos Previos

Asegúrate de tener instalado lo siguiente:

- Python 3.11 o superior
- `pip` (gestor de paquetes de Python)
- Docker y Docker Compose (opcional, para despliegue en contenedores)

### Configuración del Entorno

1.  Clona el repositorio:
    ```bash
    git clone https://github.com/PlataformaIntegradaInvestigadores/predictive_model_backend.git
    cd predictive_model_backend
    ```

2.  Crea y activa un entorno virtual (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  Instala las dependencias:
    Todas las dependencias necesarias se encuentran en el archivo `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Uso

### Ejecución en Local

Para iniciar el servidor en tu máquina local, utiliza Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

La API estará disponible en `http://localhost:8003`.

### Ejecución con Docker

El proyecto incluye un `Dockerfile` para facilitar el despliegue en contenedores.

1.  Construye la imagen de Docker:
    ```bash
    docker build -t predictive_model_backend .
    ```

2.  Ejecuta el contenedor:
    ```bash
    docker run -d -p 8003:8003 --name predictive_backend predictive_model_backend
    ```

La API estará disponible en `http://localhost:8003`.

## Endpoints de la API

La API ofrece los siguientes endpoints para interactuar con el modelo predictivo:

- **`GET /`**: Endpoint raíz que devuelve un mensaje de bienvenida.

- **`GET /api/v1/affiliations`**: Devuelve una lista completa de todos los nombres de afiliaciones disponibles para consulta.
  - **Respuesta Exitosa (200):**
    ```json
    {
      "affiliations": ["Universidad de Cuenca", "Universidad de Guayaquil", "..."]
    }
    ```

- **`GET /api/v1/projection/{affiliation_name}`**: Obtiene la proyección histórica y futura para una afiliación específica.
  - **Parámetros**:
    - `affiliation_name` (string, path): Nombre de la afiliación.
    - `projection_years` (int, query, opcional, por defecto: 5): Número de años a proyectar hacia el futuro.
    - `hypothetical_authors` (int, query, opcional): Número hipotético de autores para un análisis "What If".
  - **Respuesta Exitosa (200):**
    ```json
    {
      "affiliation_name": "Universidad de Cuenca",
      "data": [
        {"year": 2020, "publications": 150, "type": "actual"},
        {"year": 2021, "publications": 165, "type": "actual"},
        {"year": 2022, "publications": 170, "type": "predicted"},
        {"year": 2023, "publications": 175, "type": "predicted"}
      ]
    }
    ```
  - **Respuesta de Error (404):** Si la afiliación no se encuentra.
    ```json
    {"error": "La afiliación 'Nombre Inexistente' no fue encontrada."}
    ```

- **`POST /api/v1/projection/compare`**: Compara las proyecciones de una lista de afiliaciones.
  - **Body**:
    ```json
    {
      "affiliation_names": ["Universidad de Cuenca", "Universidad de Guayaquil"]
    }
    ```
  - **Parámetros**:
    - `projection_years` (int, query, opcional, por defecto: 5).
  - **Respuesta Exitosa (200):**
    ```json
    {
      "results": [
        {
          "affiliation_name": "Universidad de Cuenca",
          "data": "[...]"
        },
        {
          "affiliation_name": "Universidad de Guayaquil",
          "data": "[...]"
        }
      ]
    }
    ```

- **`GET /api/v1/ranking`**: Devuelve un ranking de afiliaciones basado en el crecimiento de publicaciones predicho para el próximo año.
  - **Respuesta Exitosa (200):**
    ```json
    {
      "ranking": [
        {
          "rank": 1,
          "affiliation_name": "Universidad de Cuenca",
          "current_year_publications": 165,
          "predicted_next_year_publications": 175,
          "growth": 10,
          "growth_percentage": 6.06
        }
      ]
    }
    ```

- **`GET /api/v1/model-details`**: Devuelve detalles y metadatos sobre el modelo de Machine Learning, incluyendo métricas de rendimiento y la importancia de las características.
  - **Respuesta Exitosa (200):**
    ```json
    {
      "model_type": "LightGBM Regressor",
      "training_data_range": "Datos históricos hasta el año 2021",
      "target_variable": "Número de publicaciones del año siguiente",
      "total_affiliations": 2708,
      "performance_metrics": {"mae": 2.20, "rmse": 4.67},
      "feature_importances": {
        "year": 120, "affiliation_encoded": 350, "..."
      }
    }
    ```

## Estructura del Repositorio

```
.
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── analytics.py   # Define los endpoints de la API
│   ├── core/
│   │   └── config.py          # Configuración de la aplicación
│   ├── models/
│   │   └── schemas.py         # Define los esquemas Pydantic para la validación de datos
│   ├── services/
│   │   └── prediction_service.py # Lógica de negocio y predicciones
│   └── main.py                # Punto de entrada de la aplicación FastAPI
├── publication_model.pkl        # Modelo de ML pre-entrenado
├── affiliation_encoder.pkl      # Codificador de etiquetas para las afiliaciones
├── publication_data.csv         # Datos históricos de publicaciones
├── preprocess_data.py           # Script para preprocesar los datos brutos (opcional)
├── requirements.txt             # Dependencias de Python
└── Dockerfile                   # Archivo para construir la imagen de Docker
```