from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
import typer

from re_cl.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_dir: Path = FIGURES_DIR,
):
    """Genera y guarda visualizaciones descriptivas del dataset procesado."""
    logger.info("Iniciando la generación de gráficos...")

    # Carga el dataset procesado
    df = pd.read_csv(input_path)

    # Asegura que el directorio de salida exista
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Gráfico de dispersión: Asistencia vs. Rendimiento ---
    # Este gráfico ayuda a visualizar la correlación entre la asistencia y el promedio general.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ASISTENCIA', y='PROM_GRAL', alpha=0.5)
    plt.title('Relación entre Asistencia y Promedio General')
    plt.xlabel('Porcentaje de Asistencia')
    plt.ylabel('Promedio General')
    scatter_path = output_dir / "asistencia_vs_rendimiento.png"
    plt.savefig(scatter_path, dpi=300) # Aumentar resolución para mejor calidad
    logger.info(f"Gráfico de dispersión guardado en {scatter_path}")
    plt.close()

    # --- 2. Gráfico de barras: Rendimiento por dependencia ---
    # Compara el rendimiento promedio entre los diferentes tipos de establecimientos.
    # Se asume la existencia de una columna 'DEPENDENCIA' que los agrupa.
    if 'DEPENDENCIA' in df.columns:
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x='PROM_GRAL', y='DEPENDENCIA', estimator=pd.Series.mean)
        plt.title('Promedio General por Tipo de Dependencia')
        bar_path = output_dir / "rendimiento_por_dependencia.png"
        plt.savefig(bar_path, dpi=300)
        logger.info(f"Gráfico de barras guardado en {bar_path}")
        plt.close()

    logger.success("Generación de gráficos completa.")


if __name__ == "__main__":
    app()
