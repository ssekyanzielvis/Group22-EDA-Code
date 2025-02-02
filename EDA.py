# Importing necessary libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from scipy import stats # type: ignore

# Load the ECG dataset
df = pd.read_csv("ecg_data.csv")

# Basic info about the dataset
print(df.info())

# Summary statistics of numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualizing the distribution of ECG signals
plt.figure(figsize=(12, 6))
plt.plot(df['ECG_signal'])  # Replace 'ECG_signal' with the actual column name
plt.title('ECG Signal over Time')
plt.xlabel('Time')
plt.ylabel('ECG Signal Amplitude')
plt.show()

# Histogram of ECG signal values
plt.figure(figsize=(10, 6))
sns.histplot(df['ECG_signal'], kde=True)
plt.title('Distribution of ECG Signal Amplitude')
plt.xlabel('ECG Signal Amplitude')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix (if applicable)
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Check for any outliers in ECG signal data
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['ECG_signal'])
plt.title('Boxplot of ECG Signal')
plt.xlabel('ECG Signal Amplitude')
plt.show()

# Apply a z-score to detect outliers
z_scores = np.abs(stats.zscore(df['ECG_signal']))
outliers = np.where(z_scores > 3)  # Typically, z-score > 3 indicates an outlier
print(f"Outliers detected at indices: {outliers}")

# Visualizing ECG signal segmentation if required (for multistrategy analysis)
# Example: Segmenting ECG data into smaller windows for further analysis
window_size = 200  # Example window size
segments = [df['ECG_signal'][i:i + window_size] for i in range(0, len(df), window_size)]
plt.figure(figsize=(12, 6))
for segment in segments[:5]:  # Plot first 5 segments
    plt.plot(segment)
plt.title('First 5 Segments of ECG Signal')
plt.xlabel('Time (within segment)')
plt.ylabel('ECG Signal Amplitude')
plt.show()

# If you have a classification column (e.g., Heart Rate Class), you can analyze it as well
if 'Heart_Rate_Class' in df.columns:
    sns.countplot(x='Heart_Rate_Class', data=df)
    plt.title('Distribution of Heart Rate Class')
    plt.xlabel('Heart Rate Class')
    plt.ylabel('Count')
    plt.show()

# Boxplot of ECG signal by class (if you have a target variable like Heart Rate Class)
if 'Heart_Rate_Class' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Heart_Rate_Class', y='ECG_signal', data=df)
    plt.title('ECG Signal by Heart Rate Class')
    plt.xlabel('Heart Rate Class')
    plt.ylabel('ECG Signal Amplitude')
    plt.show()
