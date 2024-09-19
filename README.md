# Project Overview
This project involves analyzing a credit card transaction dataset to detect fraudulent transactions. The analysis includes data preprocessing, exploratory data analysis (EDA), and the application of various machine learning models to predict fraudulent transactions.

## Libraries Used

The following libraries were used in this project:

- **pandas**: For data manipulation and analysis.
- **warnings**: To suppress warnings during the analysis.
- **matplotlib**: For data visualization.
- **seaborn**: For advanced data visualization.
- **scikit-learn**: For machine learning algorithms and data preprocessing.

## Exploratory Data Analysis (EDA)

### Loading the Data
The dataset was loaded using `pd.read_csv('creditcard.csv')`.

### Data Summary
- `df.info()`: Provided a concise summary of the DataFrame, including the number of entries, column names, data types, and memory usage.
- `df.describe()`: Offered a statistical summary of the numerical columns, including count, mean, standard deviation, min, 25th percentile, median, 75th percentile, and max values.
- `df.isnull().sum()`: Detected and counted missing values in each column.
- `df.isnull().sum().max()`: Found the maximum number of missing values across all columns.

### Correlation Analysis
Computed the correlation between each numerical column and the `Class` column, which indicates whether a transaction is fraudulent.

### Data Visualization
- **Heatmap**: Created a heatmap of the Pearson correlation coefficients for the features.
- **Count Plot**: Visualized the distribution of the `Class` column to show the imbalance between genuine and fraudulent transactions.
- **Histograms**: Displayed side-by-side histograms for the distribution of transaction amounts and transaction times.
- **Density Plot**: Visualized the distribution of transaction times for genuine and fraudulent transactions.
- **Box Plots**: Created side-by-side box plots for the distribution of transaction amounts by class.
- **KDE Plots**: Generated Kernel Density Estimate plots for each feature, comparing their distributions between genuine and fraudulent transactions.

## Data Preprocessing

### Standardization
Standardized the `Amount` and `Time` features to have a mean of 0 and a standard deviation of 1 using `StandardScaler` from scikit-learn.

### Data Splitting
Split the data into training and testing sets using `train_test_split` from scikit-learn. The target variable was `Class`, and the features were all other columns.

## Machine Learning Models

Four different classifiers were set up using scikit-learn:

1. **Logistic Regression**: `LogisticRegression()`
2. **Decision Tree Classifier**: `DecisionTreeClassifier()`
3. **K-Nearest Neighbors Classifier**: `KNeighborsClassifier()`
4. **Gaussian Mixture Model**: `GaussianMixture()`

### Model Training and Evaluation
- Each classifier was trained using the training data (`X_train` and `y_train`).
- Cross-validation scores were computed using 5-fold cross-validation to evaluate the performance of each model.

## Results

The results of the model evaluations, including accuracy, precision, recall, and F1-score, were compared to determine the best-performing model for detecting fraudulent transactions.

- **Logistic Regression**: 
    - Accuracy: 0.9991
    - Confusion Matrix: [[56854    10] [39    59]]
    - F1 Score: 0.7065868263473054

- **Decision Tree Classifier**: 
    - Accuracy: 0.9992
    - Confusion Matrix: [[56832    32] [15    83]]
    - F1 Score: 0.7487684729064039

- **K-Nearest Neighbors Classifier**: 
    - Accuracy: 0.9995
    - Confusion Matrix: [[56859     5] [22    76]]
    - F1 Score: 0.8491620111731844

### Performance Evaluation

Based on the evaluation metrics, the **K-Nearest Neighbors Classifier** performed the best overall, with the highest accuracy, confusion matrix and F1-score. Given the imbalanced nature of the dataset, precision and recall were particularly important metrics. Therefore, the K-Nearest Neighbors Classifier was selected as the final model for detecting fraudulent transactions.
