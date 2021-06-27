"""Extreme Learning Machine.

:@author: Hayden Reece Hohns
:@date: 26/03/2021
:@brief: This class is a bespoke model for solving similar problems that a MLP
neural network would solve. time series problems. It is known as an Extreme
Learning Machine (ELM) which contains two layers of weights that are randomly
initialised. The main advantage of the ELM is the copmarable predictive power
of a neural network but with very fast training times.
"""

import datetime
import os
from typing import Dict, Iterable, Union

import joblib
import numpy as np


class ELM:
    """Extreme Learning Machine model written in scikit-learn style API."""

    def __init__(
        self,
        config: Union[Dict, str, None] = None,
        directory: Union[None, str] = None,
    ) -> None:
        """Initialise all of the parameters for the ELM.

        Args:
            config (Union[Dict, str, None]): If from a previous run, you should
            pass either a dictionary of the parameters, or a string containing
            the file path to load. Otherwise they will be initialised. Defaults
            to None.
            directory (Union[None, str]): Directory to store config files and
            parameters in. Defaults to None.
        """
        self.init_time = str(datetime.datetime.now()).replace(":", "")
        if not directory:
            self.top_dir = os.getcwd()
            self.param_dir = os.path.join(self.top_dir, "params")
            print(f"Creating directories in: {self.top_dir}")
        if not os.path.exists(self.param_dir):
            print(f"Creating parameter directory at {self.param_dir}.")
            os.makedirs(self.param_dir)
        # Checks for the config (variable or string of file)
        if type(config) == str:
            self.config_path = os.path.join(self.param_dir, config)
            print(f"Reading config: {self.config_path}")
            self.algorithm_params = joblib.load(filename=self.config_path)
        if type(config) == Dict:
            # Should also check the types and keys in the dict
            self.algorithm_params = config
            self.config_path = os.path.join(
                self.param_dir, self.init_time + "-elm-params.joblib"
            )
        if config is None:
            self.algorithm_params = {
                "hidden_layer_size": 500,
                "reg_lambda": 1e-3,
                "use_max_normalisation": False,
                "use_constrained_weights": True,
            }
            self.config_path = os.path.join(
                self.param_dir, self.init_time + "-elm-params.joblib"
            )
            joblib.dump(value=self.algorithm_params, filename=self.config_path)

    def train(
        self,
        input_data: np.array,
        output_data: np.array,
        save: bool = False,
    ):
        """Train the model.

        Args:
            input_data (np.array): A 2-D array where each row is a single
            sample with potentially many columns.
            output_data (np.array): The prediction where a single value
            corresponds to a single sample of input data.
            save (bool, optional): Boolean as to whether the trained parameters
            should be saved or not. Defaults to False.
        """
        try:
            num_input_vars = input_data.shape[1]
        except IndexError:
            input_data.reshape(len(input_data), 1)
            num_input_vars = 1
        # num_output_vars = output_data.shape[1]  # number of target variables
        K_train = input_data.shape[0]  # number of training samples

        # pre-process target
        if len(output_data.shape) == 1:
            output_data = np.expand_dims(output_data, axis=-1)
        reg_lambda = self.algorithm_params["reg_lambda"]
        hidden_layer_size = self.algorithm_params["hidden_layer_size"]
        # get the input layer weights
        if self.algorithm_params["use_constrained_weights"] is False:
            # fmt: off
            W_rand = np.random.normal(
                0.0, 1.0,
                (num_input_vars, hidden_layer_size),
            )
            # fmt: on
        else:
            W_rand = np.zeros((num_input_vars, hidden_layer_size), "float32")
            for i in range(hidden_layer_size):
                Norm = 0
                while Norm < 1e-10:
                    Inds = np.random.permutation(K_train)
                    input_data_Diff = (
                        input_data[Inds[0], :] - input_data[Inds[1], :]
                    )  # noqa
                    input_data_Diff = input_data_Diff - np.mean(input_data_Diff)  # noqa
                    Norm = np.sqrt(np.sum(input_data_Diff * input_data_Diff))
                W_rand[:, i] = input_data_Diff / Norm

        # get the hidden layer activations
        A = np.maximum(-1.0, np.matmul(input_data, W_rand))

        # compute the output weights
        B = np.linalg.inv(
            np.matmul(np.transpose(A), A)
            + reg_lambda * np.identity(hidden_layer_size)  # noqa
        )
        W_outputs = np.matmul(np.matmul(np.transpose(output_data), A), B)

        self.algorithm_params["W_randoms"] = W_rand
        self.algorithm_params["W_learned"] = W_outputs

        if save:
            self.save()

    def predict(self, input_variables: Iterable) -> float:
        """Make prediction using an inner product and hidden layer.

        Args:
            input_variables (Iterable): The 'row' of data which likely contains
            floats or ints.
        Returns:
            float: A single numerical variable for the prediction.
        """
        W_randoms = self.algorithm_params["W_randoms"]
        W_learned = self.algorithm_params["W_learned"]
        a = np.maximum(-1.0, np.matmul(input_variables, W_randoms))  # size KxM
        prediction = np.matmul(a, np.transpose(W_learned))
        return prediction.tolist()[0][0]

    def save(self):
        """Dump the algorithm parameters into the config path."""
        joblib.dump(value=self.algorithm_params, filename=self.config_path)
