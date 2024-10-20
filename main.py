import pandas as pd
from src.preprocess import load_and_clean_data, add_features
from src.train_model import train_classification_model, tune_hyperparameters
from src.evaluate_model import evaluate_classification_model
import joblib

# Load and preprocess data
df = load_and_clean_data('data/Indonesian University Ranking 2020.csv')

# Add new features
df = add_features(df)

# Check available columns
print("Kolom yang tersedia dalam dataset:", df.columns)

# Categorize 'Rank'
df['Rank_Category'] = pd.cut(df['Rank'], bins=[0, 10, 50, df['Rank'].max()], labels=['Top 10', 'Top 50', 'Lainnya'])

# Encode 'University' and 'Town' columns as features
X = pd.get_dummies(df[['University', 'Town', 'Region']], drop_first=True)

# Classification target
y = df['Rank_Category']

# Train classification model
model, X_test, y_test = train_classification_model(X, y)

# Hyperparameter tuning
best_model = tune_hyperparameters(model, X, y)

# Evaluate model
accuracy = evaluate_classification_model(best_model, X_test, y_test)
print(f"Accuracy: {accuracy}")

# Save the model
joblib.dump(best_model, 'model/university_ranking_classification_model.pkl')
