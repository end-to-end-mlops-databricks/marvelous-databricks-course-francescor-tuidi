"""Module to send requests to the model serving endpoint."""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from src import logger


def send_request(records: list, endpoint: str, headers: dict) -> tuple[int, float]:
    """Send a request to the model serving endpoint and return the status code and latency.

    Args:
        records (list): List of records to send in the request.
        endpoint (str): The endpoint to send the request to.
        headers (dict): The headers to include in the request.

    Returns:
        tuple: The status code and latency of the request.
    """
    body = records
    if isinstance(records[0], list):
        body = random.choice(records)

    start_time = time.time()
    response = requests.post(
        endpoint,
        headers=headers,
        json={"dataframe_records": body},
    )
    end_time = time.time()
    latency = end_time - start_time
    status_code = response.status_code
    text = response.text

    logger.info("Response status:", status_code)
    logger.info("Response text:", text)
    logger.info("Execution time:", latency, "seconds")

    return text, status_code, latency


def send_request_concurrently(num_requests: int, records: list, endpoint: str, headers: dict) -> list:
    """Send multiple requests concurrently and calculate the average latency.

    Args:
        num_requests (int): The number of requests to send.
        records (list): List of records to send in the request.
        endpoint (str): The endpoint to send the request to.
        headers (dict): The headers to include in the request.

    Returns:
        list: List of responses from the requests.
    """

    total_start_time = time.time()
    latencies = []
    responses = []

    # Send requests concurrently
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [
            executor.submit(send_request, records=records, endpoint=endpoint, headers=headers)
            for _ in range(num_requests)
        ]

        for future in as_completed(futures):
            response, status_code, latency = future.result()
            latencies.append(latency)
            responses.append(response)

    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    # Calculate the average latency
    average_latency = sum(latencies) / len(latencies)

    logger.info("\nTotal execution time:", total_execution_time, "seconds")
    logger.info("Average latency per request:", average_latency, "seconds")

    return responses
