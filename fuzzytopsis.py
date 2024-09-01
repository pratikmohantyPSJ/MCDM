import pandas as pd
import numpy as np

# Define fuzzy mapping
fuzzy_mapping = {
    'EL': (0, 0, 0.1),
    'VL': (0, 0.1, 0.3),
    'L': (0.1, 0.3, 0.5),
    'M': (0.3, 0.5, 0.7),
    'H': (0.5, 0.7, 0.9),
    'VH': (0.7, 0.9, 1),
    'EH': (0.9, 1, 1)
}

# Load decision maker data from CSV files
dm1_df = pd.read_csv('dm1.csv')
dm2_df = pd.read_csv('dm2.csv')
dm3_df = pd.read_csv('dm3.csv')

# Load the weights CSV file
weights_df = pd.read_csv('fuzzy_topsis_DMs_criteria_wt.csv')
weights_df = weights_df.rename(columns={'Unnamed: 0': 'Criteria'})

# Convert weights DataFrame to dictionary
weights_dict = weights_df.set_index('Criteria').T.to_dict('list')

# Extract weights for each decision maker and map to fuzzy values
dm1_weights = {k.strip(): fuzzy_mapping[v[0].strip()] for k, v in weights_dict.items()}
dm2_weights = {k.strip(): fuzzy_mapping[v[1].strip()] for k, v in weights_dict.items()}
dm3_weights = {k.strip(): fuzzy_mapping[v[2].strip()] for k, v in weights_dict.items()}

# Combine weights from all decision makers
combined_weights = {}
for key in dm1_weights.keys():
    min_w = min(dm1_weights[key][0], dm2_weights[key][0], dm3_weights[key][0])
    avg_w = sum([dm1_weights[key][1], dm2_weights[key][1], dm3_weights[key][1]]) / 3
    max_w = max(dm1_weights[key][2], dm2_weights[key][2], dm3_weights[key][2])
    combined_weights[key] = (min_w, avg_w, max_w)

# Define beneficial and non-beneficial criteria
beneficial_criteria = ['exp', 'tot bugs asgn', 'new bugs', 'tot fixed bugs']
non_beneficial_criteria = ['avg fixing time(days)']

# Function to transform decision maker data
def transform_dm_data(dm_df):
    transformed_data = {}
    for col in dm_df.columns:
        col_name = col.strip()
        if col_name != 'Unnamed: 0':
            transformed_data[col_name] = [fuzzy_mapping[val.strip()] for val in dm_df[col]]
    return pd.DataFrame(transformed_data)

# Transform each decision maker's data
dm1_transformed = transform_dm_data(dm1_df.iloc[:, 1:])
dm2_transformed = transform_dm_data(dm2_df.iloc[:, 1:])
dm3_transformed = transform_dm_data(dm3_df.iloc[:, 1:])

# Combine decision matrices
def combine_decision_matrices(matrices):
    combined_matrix = {}
    for col in matrices[0].columns:
        combined_col = []
        for i in range(len(matrices[0][col])):
            min_a = min([matrix[col][i][0] for matrix in matrices])
            avg_b = sum([matrix[col][i][1] for matrix in matrices]) / len(matrices)
            max_c = max([matrix[col][i][2] for matrix in matrices])
            combined_col.append((min_a, avg_b, max_c))
        combined_matrix[col] = combined_col
    return pd.DataFrame(combined_matrix)

# Combine transformed decision matrices
combined_matrix_df = combine_decision_matrices([dm1_transformed, dm2_transformed, dm3_transformed])

# Function to normalize the fuzzy decision matrix
def normalize_fuzzy_matrix(df, beneficial_criteria, non_beneficial_criteria):
    normalized_matrix = df.copy()
    for col in df.columns:
        if col in beneficial_criteria:
            col_max = max([max(x) for x in df[col]])
            if col_max != 0:
                for i in range(len(df[col])):
                    a, b, c = df[col][i]
                    normalized_matrix[col][i] = tuple(x / col_max for x in (a, b, c))
            else:
                for i in range(len(df[col])):
                    normalized_matrix[col][i] = (0, 0, 0)
        elif col in non_beneficial_criteria:
            col_min = min([min(x) for x in df[col]])
            for i in range(len(df[col])):
                a, b, c = df[col][i]
                normalized_matrix[col][i] = tuple(col_min / x if x != 0 else 0 for x in (a, b, c))
    return normalized_matrix

# Normalize the fuzzy decision matrix
normalized_df = normalize_fuzzy_matrix(combined_matrix_df, beneficial_criteria, non_beneficial_criteria)

# Rename the column in normalized_df to match combined_weights
normalized_df = normalized_df.rename(columns={'avg fixing time': 'avg fixing time(days)'})

# Print column names and keys to verify consistency
print("Column names in normalized_df:", normalized_df.columns)
print("Keys in combined_weights:", combined_weights.keys())

# Function to apply weights
def apply_weights(normalized_df, combined_weights):
    weighted_matrix = {}
    normalized_df.columns = [col.strip() for col in normalized_df.columns]  # Strip whitespace from column names
    combined_weights = {k.strip(): v for k, v in combined_weights.items()}  # Strip whitespace from keys

    for col in normalized_df.columns:
        col_name = col.strip()  # Ensure column name is stripped
        if col_name in combined_weights:
            weighted_matrix[col_name] = []
            for i in range(len(normalized_df[col])):
                normalized_value = normalized_df[col][i]
                combined_weight = combined_weights[col_name]
                weighted_value = tuple(a * b for a, b in zip(normalized_value, combined_weight))
                print(f'Normalized value for {col_name} (row {i}): {normalized_value}')  # Debugging line
                print(f'Combined weight for {col_name}: {combined_weight}')  # Debugging line
                print(f'Weighted value for {col_name} (row {i}): {weighted_value}')  # Debugging line
                weighted_matrix[col_name].append(weighted_value)
        else:
            print(f"Warning: Column '{col_name}' not found in combined_weights")  # Debugging line
    return weighted_matrix

# Apply combined weights to normalized data
weighted_normalized_df = pd.DataFrame(apply_weights(normalized_df, combined_weights))

# Function to calculate FPIS and FNIS
def calculate_fpis_fnis(weighted_normalized_df):
    FPIS = []
    FNIS = []
    for col in weighted_normalized_df.columns:
        col_max = max(weighted_normalized_df[col], key=lambda x: (x[2], x[1], x[0]))
        col_min = min(weighted_normalized_df[col], key=lambda x: (x[0], x[1], x[2]))
        FPIS.append(col_max)
        FNIS.append(col_min)
    return FPIS, FNIS

# Calculate FPIS and FNIS
FPIS, FNIS = calculate_fpis_fnis(weighted_normalized_df)

# Function to calculate distances from FPIS and FNIS
def fuzzy_distance(a, b):
    return np.sqrt((1/3) * ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2))

def calculate_distances(weighted_normalized_df, FPIS, FNIS):
    distances_FPIS = []
    distances_FNIS = []
    for index, row in weighted_normalized_df.iterrows():
        dist_to_FPIS = sum(fuzzy_distance(row[col], FPIS[i]) for i, col in enumerate(weighted_normalized_df.columns))
        dist_to_FNIS = sum(fuzzy_distance(row[col], FNIS[i]) for i, col in enumerate(weighted_normalized_df.columns))
        distances_FPIS.append(dist_to_FPIS)
        distances_FNIS.append(dist_to_FNIS)
    return distances_FPIS, distances_FNIS

# Calculate distances from FPIS and FNIS
distances_FPIS, distances_FNIS = calculate_distances(weighted_normalized_df, FPIS, FNIS)

# Calculate closeness coefficient for each alternative
closeness_coefficients = [d_n / (d_p + d_n) for d_p, d_n in zip(distances_FPIS, distances_FNIS)]

# Rank the alternatives based on closeness coefficient
ranks = np.argsort(closeness_coefficients)[::-1]  # Higher closeness coefficient is better

# Print results
print("Combined/Aggregated Decision Matrix:")
print(combined_matrix_df)
print()

print("Normalized Fuzzy Decision Matrix:")
print(pd.DataFrame(normalized_df))
print()

print("Combined Weights Matrix:")
print(combined_weights)
print()

print("Weighted Normalized Fuzzy Decision Matrix:")
print(weighted_normalized_df)
print()

print("FPIS:", FPIS)
print("FNIS:", FNIS)
print()

print("Distances to FPIS:", distances_FPIS)
print("Distances to FNIS:", distances_FNIS)
print()

print("Closeness Coefficients:", closeness_coefficients)
print()

print("Ranks:", ranks)



import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Example classifier (replace with your model)
from sklearn.ensemble import RandomForestClassifier

# Example dataset (replace with your data)
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Initialize classifier
model = RandomForestClassifier()

# Define k-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=1)

# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Append metrics to lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Print the average of the metrics
print(f'Accuracy: {np.mean(accuracy_scores):.4f}')
print(f'Precision: {np.mean(precision_scores):.4f}')
print(f'Recall: {np.mean(recall_scores):.4f}')
print(f'F1 Score: {np.mean(f1_scores):.4f}')