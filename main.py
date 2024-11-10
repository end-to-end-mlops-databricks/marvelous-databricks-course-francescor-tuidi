from src import logger
from src.video_game_sales.config import ProjectConfig
from src.video_game_sales.data_processor import DataProcessor

config = ProjectConfig.from_yaml()
logger.info("Configuration loaded successfully.")
logger.info(config)

data_processor = DataProcessor(config)
data_processor.preprocess_data()
X_train, X_test, y_train, y_test = data_processor.split_data()