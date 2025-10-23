import pandas as pd

# Load the file you uploaded
df = pd.read_csv('hospital_appointment.csv')

# Print the initial information for data inspection
print("--- Data Info ---")
print(df.info())

# Start cleaning: Convert date columns
df['scheduling_date'] = pd.to_datetime(df['scheduling_date'], format='%d-%m-%Y')
df['appointment_date'] = pd.to_datetime(df['appointment_date'], format='%d-%m-%Y')

# Check for missing values
print("\n--- Missing Value Count ---")
print(df.isnull().sum())
duplicate_count = df.duplicated().sum()

print(f"Total number of duplicate rows found: {duplicate_count}")

# Optionally, view the first few duplicate rows
# The `keep=False` flag marks ALL copies of a duplicate row as True
if duplicate_count > 0:
    print("\n--- First 5 Duplicate Rows (All Copies Marked) ---")
    duplicate_rows = df[df.duplicated(keep=False)].head()
    print(duplicate_rows)
    
# --- 2. Remove Duplicates ---

if duplicate_count > 0:
    # Use .drop_duplicates() to remove the duplicate rows.
    # By default, it keeps the first occurrence and removes the subsequent ones (keep='first').
    df_cleaned = df.drop_duplicates(keep='first')
    
    # Verify the size reduction
    rows_removed = len(df) - len(df_cleaned)
    print(f"\nTotal rows removed: {rows_removed}")
    print(f"New DataFrame size: {len(df_cleaned)} rows")
    
    # Update the main DataFrame variable
    df = df_cleaned
else:
    print("\nNo duplicates found. The DataFrame remains unchanged.")
    import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a consistent style for the plots
sns.set_style("whitegrid")

# 1. Load and Clean Data
file_name = 'hospital_appointment.csv'
df = pd.read_csv(file_name)

# --- Initial Cleaning Steps (Date Conversion & Duplicates) ---

# Convert date columns to datetime objects
date_format = '%d-%m-%Y'
df['scheduling_date'] = pd.to_datetime(df['scheduling_date'], format=date_format, errors='coerce')
df['appointment_date'] = pd.to_datetime(df['appointment_date'], format=date_format, errors='coerce')

# Drop duplicates
df = df.drop_duplicates()

# --- 2. Visualization 1: Appointment Status Distribution (Bar Chart) ---

plt.figure(figsize=(10, 6))
status_counts = df['status'].value_counts().sort_values(ascending=False)
sns.barplot(x=status_counts.index, y=status_counts.values, palette='viridis')
plt.title('Distribution of Appointment Status', fontsize=16)
plt.xlabel('Appointment Status', fontsize=12)
plt.ylabel('Number of Appointments', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('appointment_status_distribution.png')
plt.close()

# --- 3. Visualization 2: Age Distribution (Histogram) ---

plt.figure(figsize=(10, 6))
# Remove potential negative or zero ages if present
df_age = df[df['age'] > 0]['age']
sns.histplot(df_age, bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Patients', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('age_distribution_histogram.png')
plt.close()

# --- 4. Visualization 3: Appointments Scheduled Over Time (Line Plot) ---

# Aggregate appointments by scheduling date
daily_scheduled = df.groupby(df['scheduling_date'].dt.date)['appointment_id'].count()

plt.figure(figsize=(14, 7))
daily_scheduled.plot(kind='line', color='darkorange')
plt.title('Number of Appointments Scheduled Over Time (Daily)', fontsize=16)
plt.xlabel('Scheduling Date', fontsize=12)
plt.ylabel('Number of Appointments', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('appointments_over_time.png')
plt.close()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Load and Prepare Data ---
print("--- Starting Regression Analysis ---")
df = pd.read_csv('hospital_appointment.csv')

# Drop duplicates for a cleaner dataset
df = df.drop_duplicates()

# Focus only on rows that have recorded waiting times (i.e., attended appointments)
# Also, ensure 'waiting_time' is numeric and handle potential errors during conversion
df['waiting_time'] = pd.to_numeric(df['waiting_time'], errors='coerce')
df_model = df.dropna(subset=['waiting_time'])

print(f"Data points used for model training (non-null waiting time): {len(df_model)}")

# --- 2. Feature Selection ---

# Select features (X) and target (y)
features = ['age', 'sex', 'scheduling_interval']
X = df_model[features]
y = df_model['waiting_time']

# --- 3. Data Preprocessing (One-Hot Encoding) ---

# Convert the categorical feature 'sex' into numerical format
X = pd.get_dummies(X, columns=['sex'], drop_first=True)

# Remove age 0 or negative if they exist, as they can skew the model
X = X[X['age'] > 0]
y = y[X.index] # Align the target variable y after filtering X

# --- 4. Split Data ---

# Split data into training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")

# --- 5. Model Training ---

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# --- 6. Prediction and Evaluation ---

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.3f} minutes")
print(f"R-squared (R2) Score: {r2:.3f}")

# --- 7. Coefficient Analysis (Interpretation) ---
print("\n--- Model Coefficients (Feature Importance) ---")
coefficients = pd.Series(model.coef_, index=X_train.columns)
print(coefficients)

print("\nInterpretation:")
print(f"- **{coefficients.index[0]}**: For every one year increase in age, the waiting time is expected to change by {coefficients[0]:.3f} minutes.")
print(f"- **{coefficients.index[1]}**: For every one unit increase in the scheduling interval, the waiting time is expected to change by {coefficients[1]:.3f} minutes.")
print(f"- **{coefficients.index[2]}**: The difference in waiting time for a Male patient (relative to the dropped 'Female' category) is {coefficients[2]:.3f} minutes.")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Used for Classification
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    root_mean_squared_error as rmse # RMSE is a regression metric, but we'll show its usage
)
from sklearn.preprocessing import LabelEncoder

# --- 1. Load and Prepare Data ---
print("--- Starting Classification Model & Metric Calculation ---")
df = pd.read_csv('hospital_appointment.csv')
df = df.drop_duplicates()

# --- 2. Define the Classification Problem ---

# Filter data to only include the two main outcomes (Attended vs. Did Not Attend)
df_class = df[df['status'].isin(['attended', 'did not attend'])].copy()

# The target variable (y) must be converted to 0s and 1s for the model
# 1 = 'attended' (Positive Class)
# 0 = 'did not attend' (Negative Class)
df_class['attended'] = np.where(df_class['status'] == 'attended', 1, 0)

# --- 3. Feature Selection & Preprocessing ---

# Features (X) for prediction: Age, Sex, Scheduling Interval
X = df_class[['age', 'sex', 'scheduling_interval']]
y = df_class['attended']

# Handle categorical features (Sex)
X = pd.get_dummies(X, columns=['sex'], drop_first=True)

# Remove age 0 or negative
X = X[X['age'] > 0]
y = y[X.index] # Align y after filtering X

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. Model Training (Logistic Regression) ---
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# --- 6. Classification Metrics (Precision, Recall, F1-Score) ---

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Classification Metrics (Predicting 'Attended') ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# --- 7. Regression Metric (RMSE) ---

# RMSE is not the standard metric for this classification task, 
# but it can be calculated on the predicted probabilities (the distance from 0 or 1).
# For demonstration, we use a simple placeholder for a potential regression target (like waiting_time)
# Note: For a proper RMSE calculation, you should use the Regression code provided previously.

# To demonstrate the RMSE function usage, we will calculate the error of the *predicted waiting time*
# by using a mock waiting time target, assuming you had trained a regression model previously.
# Since we trained a classification model, we'll calculate the RMSE on the classification error
# (the difference between predicted class (0 or 1) and true class (0 or 1)).

rmse_error = rmse(y_test, y_pred)

print("\n--- Regression Metric (RMSE) ---")
print(f"RMSE (Error of Classification Prediction): {rmse_error:.4f}")
print("\n*Note: For a meaningful RMSE, you would typically use a Regression model (like the one in the previous answer) to predict a continuous variable (like waiting_time).*")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    root_mean_squared_error as rmse
)

# --- 1. Load and Prepare Data ---
df = pd.read_csv('hospital_appointment.csv')
df = df.drop_duplicates()

# --- 2. Define the Classification Problem ---
# Target: Predict if appointment was 'attended' (1) or 'did not attend' (0)
df_class = df[df['status'].isin(['attended', 'did not attend'])].copy()
df_class['attended'] = np.where(df_class['status'] == 'attended', 1, 0)

# --- 3. Feature Selection & Preprocessing ---
X = df_class[['age', 'sex', 'scheduling_interval']]
y = df_class['attended']

# Handle categorical features (Sex)
X = pd.get_dummies(X, columns=['sex'], drop_first=True)

# Align X and y by removing rows where age is <= 0
X = X[X['age'] > 0]
y = y[X.index]

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. Model Training (Logistic Regression) ---
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# --- 6. Calculate Metrics ---
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rmse_val = rmse(y_test, y_pred)

# --- 7. Summarize Findings ---
print("\n========== SUMMARY OF FINDINGS ==========")

# Print Metric Scores
print("\n--- MODEL EVALUATION SCORES (Classification: Predicting 'Attended') ---")
print("| Metric | Score |")
print("|:---|:---|")
print(f"| Precision | {precision:.4f} |")
print(f"| Recall | {recall:.4f} |")
print(f"| F1-Score | {f1:.4f} |")
print(f"| RMSE* | {rmse_val:.4f} |")
print("\n*RMSE is applied here to the error between the predicted class (0 or 1) and the true class (0 or 1).")
print("\n--- VISUAL FINDINGS (See Generated Graphs) ---")
print("1. Appointment Status Distribution: Shows the total counts of attended vs. did not attend vs. other appointment statuses.")
print("2. Age Distribution of Patients: Provides the frequency of patients across different age groups.")
print("3. Appointments Scheduled Over Time: Illustrates the daily booking trend, highlighting peak or low scheduling periods.")
print("\n===========================================")
