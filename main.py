import yaml

from src import config, logger
from src.video_game_sales.data_processor import DataProcessor

logger.info("Configuration loaded successfully.")
logger.info(yaml.dump(data=config, explicit_start=True, explicit_end=True))

data_processor = DataProcessor(config)
data_processor.preprocess_data()
X_train, X_test, y_train, y_test = data_processor.split_data()
