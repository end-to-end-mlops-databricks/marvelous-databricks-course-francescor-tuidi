# COMMAND ----------

from pyspark.sql import SparkSession

from src.video_game_sales.config import ProjectConfig
from src.video_game_sales.data_processor import DataProcessor

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------

# Load the house prices dataset
url = "/" + config.data_full_path
df = spark.read.csv(url, header=True, inferSchema=True).toPandas()

# COMMAND ----------

data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess_data()
train_set, test_set = data_processor.split_data()

# COMMAND ----------

data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
