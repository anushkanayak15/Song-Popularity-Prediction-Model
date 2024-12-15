# Song Popularity Prediction Model

## Our Data Set
Our dataset contains Spotify tracks of over 125 genres. Each track has identification features including track ID, track name, artist who performed the track, and audio features such as danceability, energy, loudness, and valence, for a total of 21 features. There are 114,000 samples in the data, and it is tabular and was cleaned ahead of time. 

The feature we focused on for our data analysis is popularity, a metric ranging from 0 to 100, with 100 being the most popular. It was calculated using an algorithm based on the total number of plays the track has had and how recent those plays are. As a result, more recent songs that are played more now have higher popularity than songs played a lot in the past, which is an implication that made predicting popularity from audio features in the dataset more difficult. 

## Overview
Our project’s objective is to predict popularity with selected features of the dataset. We used forward feature selection to find features that were most strongly associated with popularity and applied various techniques for prediction including linear and logistic regression, PCA and clustering, random forests, and neural networks. Out of these methods, our key methodology that worked the best was random forests. 

## Key Methodology - Random Forest Classifier
To address the challenge of predicting the popularity of tracks in the Spotify dataset, we employed the Random Forest Classifier, a robust ensemble learning method. This approach was selected due to the model's ability to handle complex, non-linear relationships as well as its effectiveness in classification tasks, especially when dealing with a mixture of numerical and categorical features like we had in our dataset.

In particular, random forest models employ an ensemble strategy where the model combines multiple decision trees derived from bootstrapped datasets. This inherently reduces the risk of overfitting that individual trees might exhibit, allowing the ensemble approach to enhance the model's generalization capabilities for our goal. Popularity in music can be influenced by various intertwined factors such as genre, release date, artist popularity, and more. Random Forest can capture these interactions effectively without explicit feature engineering.

This is especially noteable for our goal, as the music data contained a good amount of non-linear features, with possible noise mixed in too. Random Forest's inherent feature selection mechanism helps in mitigating the impact of such noise, focusing on the most predictive features.

## Why Random Forest was Chosen (Results)
It delivered high accuracy and AUC scores, outperforming simpler models and other non-linear models like neural networks. It also effectively handled various feature types and captured complex patterns in the data, demonstrating resilience against overfitting and noise within the dataset.

On the Validation dataset, the following metrics were scored:
Validation Accuracy: 0.8326754385964912
Validation Error Rate: 0.16732456140350882
Validation True Positive Rate (Sensitivity): 0.8576235169116345
Validation True Negative Rate (Specificity): 0.8081870328524248
Validation F1 Score: 0.8354681502566093
AUROC: 0.91

After computing 5-fold cross validation, these were the results:
Fold 1 Accuracy: 0.8312280701754386
Fold 2 Accuracy: 0.8307894736842105
Fold 3 Accuracy: 0.8342105263157895
Fold 4 Accuracy: 0.828421052631579
Fold 5 Accuracy: 0.8321417606035353
AUC per Fold: [0.9100484967697291, 0.9093450572282258, 0.9094404904413814, 0.9098238853439177, 0.9132793952882816]

This can be summarized as:
Mean Accuracy: 0.8313581766821105
Mean AUROC: 0.9103874650143071

This cross-validated model produced high mean accuracy and AUROC values that were better than our other models, indicating good performance and generalization to new data. Of course, if our model did not perform well in cross-validation, then defining certain hyperparameters (max_depth, min_samples_split, max_leaf_nodes, etc.) would have been needed to prevent overfitting. However, as mentioned previously, because of the ensemble nature of the Random Forest Models that inherently prevent overfitting, we could derive extremely good model performance using sklearn's default parameters. This simple elegance together with the fact that it had the best performance across all models used, led us to choose the Random Forest Model.

## Some limitations to the model include the following:
Limited Interpretability: While Random Forests provide feature importance scores, their ensemble nature makes it difficult to interpret individual decision paths, hindering comprehensive understanding of how predictions are made.
High Computational Demand: Training multiple decision trees requires significant computational resources and time, especially with large datasets, which can be a constraint in resource-limited environments.
Handling Imbalanced Data: Despite binarizing the target variable, Random Forests can still exhibit bias towards the majority class, potentially leading to poor performance on the minority class without proper adjustments, which requires prior knowledge or hyperparameter selection.
Dependence on Feature Engineering: The model’s effectiveness heavily relies on the quality and relevance of input features. Irrelevant or noisy features can degrade performance, necessitating meticulous feature selection and engineering.
Risk of Overfitting: Although ensemble methods reduce overfitting risks, excessive tree depth or an excessive number of trees can still cause the model to overfit the training data, compromising its generalization to unseen data.

## How to Use This Code
1. Starting with the necessary libraries, begin by importing all the required Python packages. These libraries are fundamental for data manipulation, model building, evaluation, and visualization. Specifically, pandas and numpy are used for data handling, scikit-learn for building and evaluating the machine learning model, and seaborn along with matplotlib for creating insightful visualizations. Before starting to write the model, make sure that these libraries are installed in your Python environment.

2. Once the environment is set up, proceed by loading and preparing the dataset. In our code above, we first cleaned the Spotify dataset and stored it in a DataFrame named spotify_cleaned to use for this model. Define your feature matrix X by dropping non-predictive/non-numeric columns such as 'track_id', 'artists', 'album_name', 'track_name', 'track_genre', and the target variable 'popularity'. The target variable y is extracted as the 'popularity' column from this DataFrame. This focuses the model on the relevant features that contribute to predicting popularity.

3. To simplify the prediction task and address potential class imbalance, binarize the target variable. Convert the continuous popularity scores into a binary classification problem by setting a cutoff at 35, which is the median value of the popularity feature in this context. This transformation categorizes tracks as either popular (1) or not popular (0), facilitating a more balanced classification scenario.

4. With the data prepared, the next step is to split the dataset into training and testing sets. Use an 80/20 split to allocate 80% of the data for training the model and 20% for testing its performance on unseen data. The train_test_split function from scikit-learn ensures that the split is random yet reproducible by setting a random_state.

5. Initializing and training the Random Forest Classifier is the core of this model implementation. The RandomForestClassifier from scikit-learn is instantiated with a fixed random_state to ensure reproducibility of results. The model is then trained using the training data (X_train_popularity and y_train_popularity).

6. After training, make predictions on the validation set to evaluate the model's performance. Use the predict_proba method to obtain the probability estimates for each class. For further analysis, especially for plotting the ROC curve, extract the probabilities corresponding to the positive class (1) as a separate variable.

7. Evaluating the model's performance on the training data provides insights into how well the model has learned the underlying patterns. Calculate various metrics such as the confusion matrix, accuracy, error rate, True Positive Rate (TPR), True Negative Rate (TNR), and F1 score. Visualize the confusion matrix using a heatmap for an intuitive understanding of the model's predictions versus actual values.

8. Similarly, assess the model's performance on the validation (test) data to determine its generalization capabilities. Compute the same set of metrics and visualize the confusion matrix for the test set. This comparison between training and validation performance helps identify any potential overfitting or underfitting issues.

9. To further evaluate the model's discriminatory ability, plot the Receiver Operating Characteristic (ROC) curve and calculate the Area Under the Curve (AUC). The ROC curve visualizes the trade-off between the true positive rate and false positive rate at various threshold settings, while the AUC provides a single scalar value to summarize the model's performance.

10. To ensure that the model's performance is robust and not reliant on a specific train-test split, implement 5-fold cross-validation. This technique involves partitioning the dataset into five subsets (folds), training the model on four folds, and validating it on the remaining fold. This process is repeated five times, with each fold serving as the validation set once. Stratified cross-validation maintains the class distribution across folds, providing a more reliable assessment of the model's performance.

11. After running the cross-validation, review the results to understand the model's consistency across different data splits. High and consistent AUC and accuracy scores across all folds indicate a robust and reliable model. Any significant variation might suggest that the model's performance is sensitive to the specific data it was trained on, indicating potential areas for improvement such as feature selection or hyperparameter tuning.

