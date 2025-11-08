from pathlib import Path
import pandas as pd
from loguru import logger
import typer

from re_cl.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """Carga los datos crudos, realiza una limpieza básica y los guarda en el directorio de datos procesados."""
    logger.info("Iniciando el procesamiento del dataset...")

    try:
        # Se carga el dataset crudo directamente desde la ubicación parametrizada.
        # El parámetro `decimal` es clave para interpretar correctamente los números.
        df = pd.read_csv(input_path)
        logger.info(f"Dataset cargado desde {input_path} con {len(df)} filas.")

        # --- Lógica de Limpieza ---
        # Se eliminan las filas donde 'PROM_GRAL' o 'ASISTENCIA' son nulos,
        # ya que son variables fundamentales para el análisis y el modelado.
        # Esto evita que las etapas posteriores fallen o generen resultados incorrectos.
        df.dropna(subset=['PROM_GRAL', 'ASISTENCIA'], inplace=True)
        logger.info(f"Filas después de eliminar nulos en PROM_GRAL y ASISTENCIA: {len(df)}")

        # --- Guardado de Datos Procesados ---
        # Garantizamos que la ruta de salida exista antes de persistir el dataset procesado.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.success(f"Dataset procesado y guardado en {output_path}")

    # --- Manejo de Errores ---
    except FileNotFoundError:
        logger.error(f"El archivo de entrada no fue encontrado en: {input_path}")
        # Se podría añadir un manejo más sofisticado, como intentar descargar los datos.
    except Exception as e:
        logger.error(f"Ocurrió un error durante el procesamiento: {e}")


if __name__ == "__main__":
    app()
