import time
from functools import wraps

from src import logger


def log_execution_time(description):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(description)
            start_time = time.time()  # Record start time
            result = func(*args, **kwargs)  # Execute the function
            end_time = time.time()  # Record end time
            execution_time = end_time - start_time
            logger.debug(f"Function '{func.__name__}' executed in: {execution_time:.4f} seconds")
            return result

        return wrapper

    return decorator
