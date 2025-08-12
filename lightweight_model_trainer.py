import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy  as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn import tree
seed = 42
np.random.seed(seed)
dataset_path = 'datasets/all_datasets_rf_ts.csv'
top_n = 2 # Number of top features to use


# Load dataset 
data = pd.read_csv(dataset_path, low_memory=False)
if 'timestamp' in data.columns:
    data = data.drop(columns=['timestamp'])
data['attack'] = data['attack'].apply(lambda x: 1 if x > 0 else 0)  # Convert to binary labels
labels = data['attack'].values
label_dist = data['attack'].value_counts()
print("Label distribution:\n", label_dist)
# Load metrics data
data_metrics = data.iloc[:, np.r_[1:14, 19:384]]
data_metrics = data_metrics.select_dtypes(include=[np.number])
data_metrics = data_metrics.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(
    data_metrics, labels, test_size=0.3, random_state=seed, stratify=labels
)


print("----------------------------------------------------------------")
print(f"Training big classifier using {X_train.shape[1]} features (metrics) and extracting feature importance:")
model = RandomForestClassifier(n_estimators=100, random_state=seed)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

# Save feature importance to CSV
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
feature_importance_path = 'artifacts/feature_importance.csv'
importance_df.to_csv(feature_importance_path, index=False)
print(f"Feature importance saved to {feature_importance_path}")

print("----------------------------------------------------------------")
print(f"Training small classifier using top {top_n} features (metrics):")
top_features = importance_df.head(top_n)['feature'].tolist()
X_train_new = X_train[top_features]
X_test_new = X_test[top_features]
model = tree.DecisionTreeClassifier(random_state=seed)
model.fit(X_train_new, y_train)
y_test_pred = model.predict(X_test_new)
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

# Save the model
model_path = 'artifacts/decision_tree.pkl'
joblib.dump(model, model_path)
print(f"Model saved to '{model_path}'")


