"""Testing module for the ELM class.

:@author: Hayden Reece Hohns
:@date: 26/06/2021
:@brief: Testing for the Extreme Learning Machine (ELM).
"""

import os
from shutil import rmtree

import numpy as np

from ..elm import ELM


def test__init():
    """Given no inputs, test that folder is created."""
    elm = ELM()
    assert(os.path.isdir(elm.param_dir))
    rmtree(elm.param_dir)


def test_train(toy_data):
    """Test the training functionality of the ELM model.

    Args:
        toy_data (pd.DataFrame): Simple sine data to test the training
        functionality.
    """
    elm = ELM()
    elm.train(
        input_data=toy_data[["input", "noise"]].values,
        output_data=toy_data["output"].values,
    )
    assert("W_learned" in elm.algorithm_params)
    isinstance(elm.algorithm_params["W_learned"], np.ndarray)
    elm.train(
        input_data=toy_data[["input", "noise"]].values,
        output_data=toy_data["output"].values,
        save=True
    )
    assert(os.path.isfile(elm.config_path))
    rmtree(elm.param_dir)


def test_predict(toy_data):
    """Test that predictions are made correctly for the ELM.

    Args:
        toy_data (pd.DataFrame): Simple sine data to test the training
        functionality.
    """
    elm = ELM()
    elm.train(
        input_data=toy_data[["input", "noise"]].values,
        output_data=toy_data["output"].values,
    )
    y_pred = elm.predict(input_variables=toy_data[["input", "noise"]].values)
    isinstance(y_pred, float)
    rmtree(elm.param_dir)


def test_save(toy_data):
    """Test save functionality by checking the config path exists.

    Args:
        toy_data (pd.DataFrame): Simple sine data to test the training
        functionality.
    """
    elm = ELM()
    elm.train(
        input_data=toy_data[["input", "noise"]].values,
        output_data=toy_data["output"].values,
    )
    elm.save()
    assert(os.path.isfile(elm.config_path))
    rmtree(elm.param_dir)
