#Predict movie genres using director's name, actors' names, title, released date.

Project's structure: 

1. Data.py
- handle invalid data points
- numericalize categorical data
- implement sentiment analysis on titles
- implement PCA to reduce dimensionality of the data

2. Models.py
- apply cross-validation to select training and testing datasets
- apply different models to achieve the best accuracy
- for each model, use gridsearch for tuning hyperparameters
