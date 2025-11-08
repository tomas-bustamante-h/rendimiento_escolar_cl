from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Carga las variables de entorno desde un archivo .env si existe.
load_dotenv()

# --- Rutas del Proyecto ---
# Se definen las rutas principales para mantener la consistencia.
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Directorios de datos
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Directorio para modelos entrenados
MODELS_DIR = PROJ_ROOT / "models"

# Directorio para reportes y figuras generadas
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Configuración de Logging ---
# Si tqdm está instalado, se integra con loguru para que los logs no interfieran
# con las barras de progreso.
# Referencia: https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
