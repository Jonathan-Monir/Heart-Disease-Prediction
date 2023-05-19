# Heart-Disease-Prediction

This project aims to analyze a heart disease dataset and develop a predictive model to identify individuals at risk of heart disease.

# Dataset
The dataset used in this project is the Cleveland Heart Disease Data Set, which is available from the UCI Machine Learning Repository. The dataset contains 13 features related to demographic information, medical history, and diagnostic test results.

# Preprocessing
The preprocessing phase involved handling missing values by utilizing a random forest model and replacing categorical nulls with the mode. We also performed label encoding and removed outliers to ensure the quality and integrity of the data.

# Feature Selection
Feature selection techniques such as VarianceThreshold, SelectKBest with chi-squared test, Lasso regularization, SelectKBest with mutual information, and SelectKBest with ANOVA F-test were employed to identify the most informative features for our predictive model.

# Data Analysis
During the data analysis phase, we drew conclusions from the dataset. We explored numeric columns, observed the differences between positive and negative heart disease cases, analyzed the standard deviations of heart disease columns, investigated the relationship between age and heart disease, examined blood pressure values, compared heart disease occurrences between genders, and assessed the association between chest pain type, age, and the likelihood of heart disease.

# Modeling
To build our predictive model, we implemented logistic regression, SVM, decision tree, and random forest algorithms. The performance of each model was evaluated based on accuracy, with logistic regression achieving 85% accuracy, SVM achieving 84% accuracy, decision tree achieving 76% accuracy, and random forest achieving 81% accuracy. Finally, we employed hyperparameter tuning techniques to optimize the models and select the best-performing model.

# Deployment
The best-performing model was deployed using Flask. The deployed model can be accessed [HERE]([https://www.google.com/](https://jonathan-monir-heart-disease-prediction-deployment-4uu7zl.streamlit.app/))


# Conclusion
In conclusion, this AI project focused on analyzing a heart disease dataset and developing a predictive model to identify individuals at risk of heart disease. Through thorough preprocessing techniques, we ensured the quality and completeness of the dataset by handling missing values, performing label encoding, and removing outliers. Feature selection methods allowed us to identify the most relevant features for our predictive model, enhancing its accuracy and interpretability.

During the data analysis phase, we drew significant insights from the dataset. Notably, we observed that positive heart disease values tended to have higher mean or median values in most positive heart disease columns. The standard deviation of negative heart disease columns was higher in certain cases, indicating that positive heart disease cases occupied lower ranges and were potentially predictable. We also examined the impact of age, gender, and chest pain type on the likelihood of heart disease, providing valuable information for further medical investigation.

By employing logistic regression, SVM, decision tree, and random forest algorithms, we achieved promising accuracies for our predictive models. Logistic regression and SVM demonstrated the highest accuracies of 85% and 84%, respectively, while decision tree and random forest achieved accuracies of 76% and 81%. Hyperparameter tuning further optimized the models, ensuring their optimal performance.

This project's findings and predictive models can assist healthcare professionals in identifying individuals at risk of heart disease. By leveraging machine learning techniques, we can provide valuable insights to support early detection, prevention, and personalized treatment strategies for heart disease patients. Future work may involve exploring additional algorithms, incorporating more comprehensive datasets, and collaborating with domain experts to enhance the accuracy and clinical utility of our predictive models.
