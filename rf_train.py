
# Importing libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("bangladesh_student_performance_2018.csv")

# Convert Pandas string types to objects
for col in df.select_dtypes(include=['string', 'object']).columns:
    df[col] = df[col].astype(object)


# Feature Engineering
if 'date' in df.columns:
    df.drop(columns=['date'], inplace=True)

# Combine Parental Education
df['Total_Parent_Edu'] = df['M_Edu'] + df['F_Edu']

# Log Transform Tuition Fee (Variance Handling)
df['tuition_fee'] = np.log1p(df['tuition_fee'])

# Target and features
X = df.drop('hsc_result', axis=1)
y = df['hsc_result']


# Column Split  
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(include=['object']).columns


# Preprocessing Pipeline
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])


# XGBoost Model 
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)


# Full Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the model
full_pipeline.fit(X_train, y_train)


# Evaluation
y_pred = full_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Training Complete!")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")


# Save model
with open("student_rf_pipeline.pkl", "wb") as f:
    pickle.dump(full_pipeline, f)
 
print("Model saved as student_rf_pipeline.pkl")


# For XAI integration:
# Extract feature names after OneHotEncoding
ohe_features = full_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, ohe_features])

# Get importance scores from XGBoost
importances = full_pipeline.named_steps['model'].feature_importances_

# Save both names and scores to a file
with open("feature_importance.pkl", "wb") as f:
    pickle.dump({"names": all_feature_names, "scores": importances}, f)

print("Feature importance data saved!")