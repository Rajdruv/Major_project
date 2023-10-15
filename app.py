from model_training.time_series_model.convert_log_csv import convert_main
from model_training.time_series_model.train_timeseries import train_main

from model_training.transformer_model.train_transformer import train_transformer_main


def train_all_model():
    # convert_main()
    # train_main()
    train_transformer_main()


if __name__ == "__main__":
    train_all_model()
