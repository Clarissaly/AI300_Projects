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
