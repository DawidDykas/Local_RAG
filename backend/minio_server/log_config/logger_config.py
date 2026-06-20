import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("global_logger")
logger.setLevel(logging.INFO)

# Konsola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Plik (opcjonalnie z rotacją)
file_handler = RotatingFileHandler("backend.log", maxBytes=5*1024*1024, backupCount=3)
file_handler.setLevel(logging.DEBUG)  # zapisujemy wszystko do pliku
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

