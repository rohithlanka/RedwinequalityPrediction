# Red Wine Quality Prediction: Machine Learning Classification

## Project Overview
This project focuses on predicting the quality of red wine based on physicochemical properties using machine learning techniques. The dataset consists of attributes such as acidity, sugar content, alcohol level, and more. The quality of wine is predicted by training a variety of models, including Support Vector Machines (SVC), AdaBoost, Logistic Regression, Decision Trees, and Random Forest classifiers. This project demonstrates data preprocessing, feature engineering, model evaluation, and hyperparameter tuning using GridSearchCV to improve prediction accuracy.

## Key Technical Highlights
- **Data Preprocessing**: The dataset is loaded, cleaned, and normalized. Duplicate values are removed, and the `quality` feature is encoded as a binary classification (greater than 5 as `1`, else `0`).
  
- **Feature Normalization**: Applied normalization techniques on the features using Scikit-learn's `normalize()` function, improving model convergence during training.

- **Modeling and Evaluation**:
  - **Benchmark Model**: Started with a Support Vector Classifier (SVC) to establish a baseline model.
  - **Multiple Classifiers**: Implemented a variety of classification algorithms including:
    - **AdaBoost** for boosting weak classifiers.
    - **Logistic Regression** for linear classification.
    - **Decision Trees** for non-linear decision boundaries.
    - **Random Forest** for ensemble learning, aggregating predictions from multiple decision trees.
  
- **Hyperparameter Tuning**: Used GridSearchCV to optimize Random Forest parameters (`n_estimators`, `max_features`, `criterion`), enhancing the model’s performance.

- **Model Comparison**: Accuracy scores for each model are compared visually using bar plots, demonstrating the effect of hyperparameter optimization and the difference in accuracy across models.


**Key Insights**:
- Optimizing hyperparameters with GridSearchCV improves the accuracy significantly, especially for the Random Forest model.
- The comparison of unoptimized vs. optimized models shows how fine-tuning parameters enhances model predictions.

## Detailed Steps

### 1. Data Loading and Preprocessing
The dataset is imported and inspected. Basic statistics such as number of rows and columns, and value counts for `quality` are displayed. Duplicate values are removed to ensure data quality. The `quality` attribute is encoded into binary values (`1` for quality > 5 and `0` otherwise) for binary classification.

### 2. Normalization of Features
The features are normalized to bring all values into a similar scale. This prevents bias towards variables with larger numeric ranges during model training.

### 3. Splitting the Dataset
The dataset is split into training and testing sets (80% training, 20% testing), ensuring that models can be trained and validated on separate data.

### 4. Model Training and Evaluation
Several classifiers are trained on the dataset:
- **SVC (Support Vector Classifier)** is applied as the benchmark.
- **AdaBoost**, **Logistic Regression**, **Decision Trees**, and **Random Forest** models are tested and compared using accuracy scores.
- A Random Forest model is then optimized using **GridSearchCV**, exploring various hyperparameters like `n_estimators`, `max_features`, and `criterion`.

### 5. Results Visualization
The accuracy of the models before and after optimization is compared using **Matplotlib** bar plots. The Random Forest model, after tuning, achieves the highest accuracy.

## Future Enhancements
- **Deep Learning**: Explore the application of deep learning models such as Neural Networks to further improve prediction accuracy.
- **Feature Engineering**: Include additional features such as geographical information, wine production techniques, etc., to enhance the model's predictive power.
- **Real-time Deployment**: Develop a pipeline to deploy this model for real-time predictions using a web-based interface or API.

## Conclusion
This project demonstrates a robust pipeline for predicting red wine quality using a range of machine learning algorithms. The Random Forest model, optimized with GridSearchCV, offers the highest accuracy, highlighting the importance of hyperparameter tuning in machine learning. Future improvements could focus on advanced models and the inclusion of more diverse features.

