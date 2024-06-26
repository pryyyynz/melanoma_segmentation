"""
Main file for training and testing models.
The name of the function should remain the same.
They should return the results of training and testing respectively.
Using the class TrainResults and TestResults is recommended.
"""
import asyncio
import os
from typing import Any
from pymlab.train import train as pm_train 
from pymlab.test import test as pm_test
from pymlab.utils import fetch_parameters
from train_model import train_model as run_train_model
from test_model import test_model as run_test_model

def main(
    pkg_name: str,
    dataset: str,
    result_id: str,
    user_token: str,
    api_url: str,
    trained_model: str = "",
) -> Any:
    parameters = fetch_parameters(config_path=os.getcwd() + "/config.train.txt")
    if pkg_name == "pymlab.train":
        parameters = fetch_parameters(config_path=os.getcwd() + "/config.train.txt")
        asyncio.run(
            pm_train(run_train_model, result_id=result_id, dataset_path=dataset, parameters=parameters, api_url=api_url,
                     user_token=user_token))
    elif pkg_name == "pymlab.test":
        asyncio.run(
            pm_test(run_test_model, result_id=result_id, api_url=api_url, user_token=user_token, dataset_path=dataset,
                    parameters=parameters, trained_model=trained_model))
    return "hello"