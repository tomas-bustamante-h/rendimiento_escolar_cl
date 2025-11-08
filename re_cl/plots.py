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
    """Genera visualizaciones descriptivas del dataset."""
    logger.info("Iniciando la generación de gráficos...")
    df = pd.read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gráfico de dispersión: Asistencia vs. Rendimiento
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ASISTENCIA', y='PROM_GRAL', alpha=0.5)
    plt.title('Relación entre Asistencia y Promedio General')
    plt.xlabel('Porcentaje de Asistencia')
    plt.ylabel('Promedio General')
    scatter_path = output_dir / "asistencia_vs_rendimiento.png"
    plt.savefig(scatter_path)
    logger.info(f"Gráfico de dispersión guardado en {scatter_path}")
    plt.close()

    # 2. Gráfico de barras: Rendimiento por dependencia
    # Asumiendo una columna 'DEPENDENCIA' que agrupa los tipos
    if 'DEPENDENCIA' in df.columns:
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x='PROM_GRAL', y='DEPENDENCIA', estimator=pd.Series.mean)
        plt.title('Promedio General por Tipo de Dependencia')
        bar_path = output_dir / "rendimiento_por_dependencia.png"
        plt.savefig(bar_path)
        logger.info(f"Gráfico de barras guardado en {bar_path}")
        plt.close()

    logger.success("Generación de gráficos completa.")


if __name__ == "__main__":
    app()
