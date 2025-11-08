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

    # Definir variables categóricas y numéricas
    # Asumiendo que las columnas de dependencia y región ya están en el dataset
    categorical_features = [col for col in df.columns if 'COD_DEPE' in col or 'COD_REG_RBD' in col]
    categorical_features.append('GEN_ALU') # Asumiendo una columna de género
    numerical_features = ['ASISTENCIA']
    
    target = 'PROM_GRAL'

    X = df[categorical_features + numerical_features]
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un preprocesador para las variables categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

    # Crear el pipeline de modelado
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', LinearRegression())])

    logger.info("Ajustando el modelo de regresión lineal...")
    pipeline.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.success(f"Modelo entrenado y guardado en {model_path}")


if __name__ == "__main__":
    app()
