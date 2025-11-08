from pathlib import Path
import pandas as pd
import joblib
from loguru import logger
import typer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from re_cl.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """Entrena un modelo de regresión lineal para predecir el rendimiento escolar."""
    logger.info("Iniciando el entrenamiento del modelo...")

    df = pd.read_csv(input_path)
    logger.info(f"Dataset de entrenamiento cargado con {len(df)} filas.")

    # --- 1. Definición de Variables ---
    # Se definen las variables predictoras (features) y la variable objetivo (target).
    categorical_features = [col for col in df.columns if 'COD_DEPE' in col or 'COD_REG_RBD' in col]
    if 'GEN_ALU' in df.columns:
        categorical_features.append('GEN_ALU')
    numerical_features = ['ASISTENCIA']
    target = 'PROM_GRAL'

    X = df[categorical_features + numerical_features]
    y = df[target]

    # --- 2. División de Datos ---
    # Se divide el dataset en conjuntos de entrenamiento y prueba.
    # En este caso, solo usamos el de entrenamiento para ajustar el modelo final.
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Preprocesamiento y Pipeline ---
    # Se crea un transformador para aplicar One-Hot Encoding a las variables categóricas.
    # `handle_unknown='ignore'` es útil si el modelo en producción encuentra categorías no vistas.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

    # Se define el pipeline que primero preprocesa los datos y luego ajusta el regresor.
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', LinearRegression())])

    # --- 4. Entrenamiento y Guardado ---
    logger.info("Ajustando el modelo de regresión lineal...")
    pipeline.fit(X_train, y_train)

    # Se asegura de que el directorio del modelo exista y guarda el pipeline entrenado.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.success(f"Modelo entrenado y guardado en {model_path}")


if __name__ == "__main__":
    app()
