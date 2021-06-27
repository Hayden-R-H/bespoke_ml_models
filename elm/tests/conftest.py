"""Define all variables necessary for unit testing."""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def toy_data():
    """Toy data for testing."""
    sine_data = np.sin(np.arange(start=0, stop=1, step=0.001))
    noise_data = np.random.normal(loc=0, scale=0.01, size=len(sine_data))
    df = pd.DataFrame(data={"input": sine_data, "noise": noise_data})
    df["output"] = df["input"].shift(-1)
    return df.dropna()
