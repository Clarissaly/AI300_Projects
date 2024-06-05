df_transformed = df.copy()

# label encoding (binary variables)
label_encoding_columns = ['churn_label', 'gender', 'has_internet_service', 'has_unlimited_data', 'has_phone_service', 'has_multiple_lines', 'has_premium_tech_support','has_online_security', 'has_online_backup', 'has_device_protection', 'paperless_billing', 'stream_tv', 'stream_movie', 'stream_music', 'senior_citizen', 'married']

# encode categorical binary features using label encoding
for column in label_encoding_columns:
    if column == 'gender':
        df_transformed[column] = df_transformed[column].map({'Female': 1, 'Male': 0})
    else: 
        df_transformed[column] = df_transformed[column].map({'Yes': 1, 'No': 0}) 



# one-hot encoding (categorical variables with more than two levels)
one_hot_encoding_columns = ['internet_type', 'contract_type', 'payment_method']

# encode categorical variables with more than two levels using one-hot encoding
df_transformed = pd.get_dummies(df_transformed, columns = one_hot_encoding_columns)


# min-max normalization (numeric variables)
min_max_columns = ['tenure_months', 'total_monthly_fee', 'total_charges_quarter']

# scale numerical variables using min max scaler
for column in min_max_columns:
        # minimum value of the column
        min_column = df_transformed[column].min()
        # maximum value of the column
        max_column = df_transformed[column].max()
        # min max scaler
        df_transformed[column] = (df_transformed[column] - min_column) / (max_column - min_column)

# Add here the ML models tested

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# Split the transformed data into features and target variable
X = df_transformed.drop('churn_label', axis=1)
y = df_transformed['churn_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# XGBoost
import xgboost as xgb

# Create an XGBoost model
model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)
print("XGBoost AUC Score:", auc_score)

# CatBoost
from catboost import CatBoostClassifier
# Create a CatBoost model
model = CatBoostClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)
print("CatBoost AUC Score:", auc_score)

# LightGBM
import lightgbm as lgb

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Set the parameters for the model
params = {
    'objective': 'binary',
    'metric': 'auc'
}

# Train the model
model = lgb.train(params, train_data, valid_sets=[test_data])

# Make predictions
y_pred = model.predict(X_test)

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred)
print("LightGBM AUC Score:", auc_score)


# Select the best performing model and input to estimators for RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

# define the parameter grid
grid_parameters = {'n_estimators': [80, 90, 100, 110, 115, 120],
                   'max_depth': [3, 4, 5, 6],
                   'max_features': [None, 'auto', 'sqrt', 'log2'], 
                   'min_samples_split': [2, 3, 4, 5]}


# define the RandomizedSearchCV class for trying different parameter combinations
random_search = RandomizedSearchCV(estimator=GradientBoostingClassifier(),
                                   param_distributions=grid_parameters,
                                   cv=5,
                                   n_iter=150,
                                   n_jobs=-1)

# fitting the model for random search 
random_search.fit(X_train, y_train)

# print best parameter after tuning
print(random_search.best_params_)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# make the predictions
random_search_predictions = random_search.predict(X_test)

# construct the confusion matrix
confusion_matrix = confusion_matrix(y_test, random_search_predictions)

# visualize the confusion matrix
confusion_matrix

# print classification report 
print(classification_report(y_test, random_search_predictions))

# print the auc score of the model after hyparameter tuning 
roc_auc_score(y_test, random_search_predictions)
