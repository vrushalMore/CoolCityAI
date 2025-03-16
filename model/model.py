import pandas as pd
from pycaret.classification import setup, compare_models, save_model

# Load the dataset
df = pd.read_csv("data.csv")  # Make sure the correct path is used

# Initialize PyCaret for binary classification
clf_setup = setup(df, target="cloud_seeding", session_id=42, normalize=True)

# Train multiple models and select the best one
best_model = compare_models()

# Save the best model in .pkl format
save_model(best_model, "best_cloud_seeding_model")
