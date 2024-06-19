# Spam_Mail_Prediction_System
This is a Spam Mail Prediction System made using Machine Learning and Python.

Spam Mail Detection System using Logistic Regression and TF-IDF

Overview
This project demonstrates a simple yet effective spam detection system using machine learning techniques. It leverages a logistic regression classifier trained on a dataset of mails categorized as 'spam' or 'ham' (not spam). The system preprocesses the text data, extracts features using TF-IDF (Term Frequency-Inverse Document Frequency), and evaluates its performance on both training and test datasets.

Dependencies
- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For machine learning models and evaluation metrics.

Steps

1. Data Collection & Pre-Processing
   - Loads SMS data from a CSV file into a Pandas DataFrame.
   - Handles missing values by replacing them with empty strings.
   - Encodes labels ('spam' as 0 and 'ham' as 1).

2. Splitting Data
   - Splits the dataset into training and testing sets using `train_test_split` from scikit-learn.

3. Feature Extraction
   - Uses `TfidfVectorizer` from scikit-learn to convert text data into numerical feature vectors.

4. Model Training
   - Utilizes a logistic regression model (`LogisticRegression` from scikit-learn) to train on the training data.

5. Model Evaluation
   - Computes accuracy scores for both training and test datasets to evaluate model performance.

6. Prediction
   - Implements a predictive system where new SMS messages can be classified as 'spam' or 'ham'.

Key Files
- dataset.csv: Contains the mails dataset.
- Spam_Mail_Prediction_System.ipynb: Script implementing the spam detection system.

Usage
- Ensure Python dependencies (`numpy`, `pandas`, `scikit-learn`) are installed.
- Run `spam_detection.py` to train the model, evaluate its performance, and test with new messages.

Results
- Achieved an accuracy of approximately 96.7% on the training dataset and 96.6% on the test dataset, indicating robust performance in distinguishing between spam and non-spam mails.

Conclusion
This project serves as a practical example of using machine learning for text classification tasks, specifically spam detection, showcasing the application of logistic regression and TF-IDF in natural language processing.
