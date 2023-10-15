import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

import pickle

with open("./best_model.np", "rb") as model_file:
    model = pickle.load(model_file)
