import pandas as pd

#load training data
def load_training_data(data_path):
    """
    Loads historical claims dataset.
    """

    df = pd.read_csv(data_path)

    return df

#load scoring data
def load_scoring_data(data_path):
    """
    Loads current claims dataset for scoring.
    """

    df = pd.read_csv(data_path)

    return df