# Indonesia University Ranking ML

## Description
The Indonesia University Ranking ML project aims to classify universities in Indonesia based on their rankings using machine learning techniques. Utilizing data from the 2020 Indonesian University Rankings, the project categorizes universities into various tiers (e.g., Top 10, Top 50, Others) based on their ranking. By implementing a Random Forest Classifier and enhancing the dataset with additional features, the model achieves an impressive accuracy of over 99%.

## Key Features
- **Data Preprocessing**: Cleaned and prepared the dataset, handled missing values, and encoded categorical features.
- **Feature Engineering**: Added new features such as regional categorization to improve model performance.
- **Model Training and Evaluation**: Utilized a Random Forest Classifier, tuned hyperparameters, and achieved high accuracy.
- **Model Saving**: Saved the trained model for future use and deployment.
- **Interactive Notebooks**: Documented the data preprocessing and model training process in Jupyter notebooks for transparency and reproducibility.

## Installation
Clone the repository and install the required dependencies:

git clone https://github.com/haekalsetiawan/Indonesia-University-Ranking-ML.git

cd Indonesia-University-Ranking-ML

pip install -r requirements.txt

## Usage
Preprocess the data:

python src/preprocess.py

Train the model:

python src/train_model.py

Evaluate the model:

python src/evaluate_model.py

## Contributor
Haekal Setiawan (@haekalsetiawan)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
