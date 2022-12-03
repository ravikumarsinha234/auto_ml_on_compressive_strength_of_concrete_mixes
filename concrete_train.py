import pycaret.regression as carreg
from time import perf_counter
import pandas as pd

# How many models we will tune
n_select = 6

# Read in the training data
train_df = pd.read_csv("train_concrete.csv")

# Set up the regression experiment session
print("*** Setting up session***")
t1 = perf_counter()
exp1 = carreg.setup(data=train_df, target="csMPa")
t2 = perf_counter()
print(f"*** Set up: {t2 - t1:.2f} seconds")

# Do a basic comparison of models (no turbo)
best = carreg.compare_models(n_select=6, turbo=False)
t3 = perf_counter()
print(f"*** compare_models: {t3 - t2:.2f} seconds")

# List the best models
print(f"*** Best:")
for b in best:
    print(f"\t{b.__class__.__name__}")

# Go through the list of models
for i, model in enumerate(best):
    # Tune the model (try 24 parameter combinations)
    print(f"*** {i} - {model.__class__.__name__} ***")
    # print('Fitting 10 folds for each of 24 candidates, totalling 240 fits')
    tuned_model = carreg.tune_model(model, n_iter=24)

    # Finalize the model
    final_model = carreg.finalize_model(tuned_model)

    # Save the model
    carreg.save_model(final_model, model.__class__.__name__)

t4 = perf_counter()
print(f"*** Tuning and finalizing: {t4 - t3:.2f} seconds")
print(f"*** Total time: {t4 - t1:.2f} seconds")
