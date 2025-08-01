# Your First Step with Auto-Sklong

# Your First Step with Auto-Sklong

Let's walk you though your first usage of Auto-Sklong, the AutoML system for longitudinal data. 
This guide will help you set up a basic AutoML run using the `GamaLongitudinalClassifier` from the GAMA library,
which is designed to handle temporal dependencies in longitudinal datasets.

**Have Read Beforehand:**

- [ ] [Temporal Dependency Guide](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/)
- [ ] [Data Format Tutorial](https://scikit-longitudinal.readthedocs.io/latest/tutorials/sklong_longitudinal_data_format/)

**Tech Prerequisites:**

- [ ] Install dependencies: `pip install scikit-longitudinal`
- [ ] Ensure your data is in CSV format with columns representing features across time waves and a target column. Please
  look at
  the [Sklong wide/long format guide](https://scikit-longitudinal.readthedocs.io/latest/tutorials/sklong_longitudinal_data_format/)
  for more details.

## 🔍 Auto-Sklong: Explore Your First AutoML Run

!!! important "Dataset Used in Tutorials"
    The tutorials use a synthetic dataset mimicking health-related longitudinal data. It's generated for illustrative
    purposes and does not represent real-world data.
    
    ??? note "Dataset Generation Code"
        ```python
        import pandas as pd
        import numpy as np
    
        n_rows = 500
    
        columns = [
        'age', 'gender',
        'smoke_w1', 'smoke_w2',
        'cholesterol_w1', 'cholesterol_w2',
        'blood_pressure_w1', 'blood_pressure_w2',
        'diabetes_w1', 'diabetes_w2',
        'exercise_w1', 'exercise_w2',
        'obesity_w1', 'obesity_w2',
        'stroke_w2'
        ]
    
        data = []
    
        for i in range(n_rows):
        row = {}
        row['age'] = np.random.randint(40, 71)  
        row['gender'] = np.random.choice([0, 1])  
                
        for feature in ['smoke', 'cholesterol', 'blood_pressure', 'diabetes', 'exercise', 'obesity']:
        w1 = np.random.choice([0, 1], p=[0.7, 0.3])
        if w1 == 1:
        w2 = np.random.choice([0, 1], p=[0.2, 0.8])  
        else:
        w2 = np.random.choice([0, 1], p=[0.9, 0.1])  
        row[f'{feature}_w1'] = w1
        row[f'{feature}_w2'] = w2
                
        if row['smoke_w2'] == 1 or row['cholesterol_w2'] == 1 or row['blood_pressure_w2'] == 1:
        p_stroke = 0.2  
        else:
        p_stroke = 0.05  
        row['stroke_w2'] = np.random.choice([0, 1], p=[1 - p_stroke, p_stroke])
                
        data.append(row)
    
        # Create DataFrame
        df = pd.DataFrame(data)
    
        # Save to a new CSV file
        csv_file = './extended_stroke_longitudinal.csv'
        df.to_csv(csv_file, index=False)
        print(f"Extended CSV file '{csv_file}' created successfully.")
        ```
    
    The dataset looks like:
    
    | age | gender | smoke_w1 | smoke_w2 | cholesterol_w1 | cholesterol_w2 | blood_pressure_w1 | blood_pressure_w2 | diabetes_w1 | diabetes_w2 | exercise_w1 | exercise_w2 | obesity_w1 | obesity_w2 | stroke_w2 |
    |-----|--------|----------|----------|----------------|----------------|-------------------|-------------------|-------------|-------------|-------------|-------------|------------|------------|-----------|
    | 66  | 0      | 0        | 1        | 0              | 0              | 0                 | 0                 | 1           | 1           | 0           | 1           | 0          | 0          | 0         |
    | 59  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 1           | 1           | 1           | 1           | 1          | 1          | 1         |
    | 63  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 0           | 0           | 0           | 0           | 0          | 0          | 1         |
    | 47  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 0           | 0           | 0           | 0           | 1          | 0          | 0         |
    | 44  | 0      | 0        | 0        | 1              | 1              | 1                 | 1                 | 0           | 0           | 0           | 0           | 1          | 1          | 1         |
    | 69  | 1      | 0        | 0        | 0              | 0              | 1                 | 1                 | 0           | 0           | 0           | 0           | 0          | 0          | 0         |
    | 63  | 0      | 0        | 0        | 0              | 0              | 0                 | 0                 | 0           | 0           | 0           | 0           | 0          | 0          | 0         |
    | 48  | 1      | 0        | 0        | 0              | 0              | 0                 | 0                 | 0           | 0           | 0           | 1           | 0          | 0          | 0         |
    | 49  | 1      | 0        | 0        | 0              | 0              | 0                 | 0                 | 0           | 0           | 0           | 1           | 0          | 1          | 0         |

## Step 1: Load and Prepare Data

Using the synthetic `extended_stroke_longitudinal.csv` from the dataset generation code above:

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./extended_stroke_longitudinal.csv')
dataset.load_data_target_train_test_split(target_column='stroke_w2', test_size=0.2, random_state=42)
dataset.setup_features_group([[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]) # Or use a preset, such as `elsa`. Read more in https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/#pre-set-features_group-and-non_longitudinal_features
```

## Step 2: Initialise and Fit the AutoML System

Use `GamaLongitudinalClassifier` to automate pipeline search, prioritizing temporal-aware models:

```python
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier
from gama.search_methods import BayesianOptimisation

automl = GamaLongitudinalClassifier(
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    max_total_time=60,  # (in seconds) Short run for tutorial; increase for real use
    scoring='roc_auc', # can chance the scoring metric to `optimise`
    search=BayesianOptimisation(), # Other options exist, explore the API reference for more details
    random_state=42 # For reproducibility
)

automl.fit(dataset.X_train, dataset.y_train)
```

## Step 3: Predict and Evaluate

```python
y_pred = automl.predict(dataset.X_test)
y_prob = automl.predict_proba(dataset.X_test)

from sklearn.metrics import accuracy_score, roc_auc_score

print(f"Accuracy: {accuracy_score(dataset.y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(dataset.y_test, y_prob[:, 1]):.4f}")

print("Classification Report:")
print(classification_report(dataset.y_test, y_pred))
```

## Step 4: Export and Reuse the Best Pipeline

```python
automl.export_script('best_autosklong_pipeline.py')
print("Best pipeline exported! Load it for production use.")
```

This introduces basic Auto-Sklong usage. Experiment with longer search times or different scorings for better results.
To understand the under the hood process, we recommend reading the [Search Space Guide](./search_space.md) as well
as obviously the paper, see more in [Publications](../publications.md).