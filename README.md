# NSL-KDD

# Introsuction


In the ever-evolving landscape of cybersecurity, where the constant threat of malicious attacks looms, the imperative task of analyzing and detecting intrusions becomes paramount for the protection of computer systems. This project delves deep into the realm of cybersecurity, specifically directing its attention towards harnessing the power of machine learning algorithms, with a particular emphasis on neural networks. The objective is to meticulously analyze and categorize various manifestations of network attacks, leveraging the comprehensive "NSL_KDD" dataset.

The journey unfolds through a series of meticulously orchestrated steps, commencing with an in-depth exploratory data analysis (EDA), progressing to intricate preprocessing methodologies, and culminating in the construction and optimization of neural network models. The overarching goal of this study is to cultivate models endowed with resilience, capable of adeptly identifying and thwarting intrusions.

Within this report, readers will find a comprehensive exposition of the adopted methodology, shedding light on the nuances of each phase. The narrative encompasses the challenges encountered during the project, offering insights into the intricacies of working within the dynamic field of cybersecurity. The results obtained are presented in meticulous detail, accompanied by a critical analysis of the performance exhibited by the developed models. This holistic approach provides a thorough understanding of the project's progression, making it an invaluable resource for those navigating the complex landscape of intrusion detection in the era of cybersecurity.

## overview of the tools and libraries employed in the subsequent data analysis and modeling processes 

In this section of the report, we establish the foundational setup for our project by importing essential libraries and configuring the environment. The following libraries play pivotal roles in data analysis, machine learning, and visualization:

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import missingno as mn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
```

1. **Library Purpose:**
   - **Pandas and NumPy:** Primarily used for data manipulation and numerical operations.
   - **TensorFlow:** Employed for building and training neural network models.
   - **Matplotlib and Seaborn:** Utilized for visualizing data and generating insightful plots.
   - **scikit-learn (sklearn):** A comprehensive machine learning library used for implementing various algorithms.
   - **Missingno:** Facilitates the visualization of missing data in the dataset.

2. **Data Preprocessing and Standardization:**
   - Data preprocessing is a crucial step, and the `StandardScaler` from scikit-learn is employed to standardize the data, ensuring it is on a normal scale.

   ```python
   # Data preprocessing and standardization
   from sklearn.preprocessing import StandardScaler
   ```

3. **Train-Test Split and Model Imports:**
   - To assess model performance, the dataset is split into training and testing sets using `train_test_split`. Additionally, classifiers including `DecisionTreeClassifier`, `RandomForestClassifier`, and `LogisticRegression` are imported for further analysis.

   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.linear_model import LogisticRegression
   ```

4. **Model Evaluation Metrics:**
   - Metrics such as `classification_report` and `confusion_matrix` from scikit-learn are essential tools for evaluating the performance of the developed models.

   ```python
   from sklearn.metrics import classification_report, confusion_matrix
   ```

5. **Suppressing Warnings:**
   - The code includes a `warnings.filterwarnings("ignore")` statement to suppress any non-essential warnings that might occur during the execution of the code.

   ```python
   import warnings
   warnings.filterwarnings("ignore")
   ```

This comprehensive setup establishes the groundwork for our subsequent data analysis, preprocessing, and model development.



```python
# Define Column Names:
# We start by defining an array of column names, 'col_names', which encapsulates the various features and attributes present in our dataset.
col_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate", "class", 'difficulty level']) 

# Load Dataset:
# We proceed to load the dataset from the "NSL_KDD.csv" file into a Pandas DataFrame ('train_df').
# The 'names' parameter is utilized to assign our predefined column names to the dataset.
train_df = pd.read_csv("NSL_KDD.csv", names = col_names)

# Display Dataset:
# We showcase a snippet of the loaded dataset using 'train_df'. This allows us to inspect the initial rows and understand the structure and content of the data.
train_df

# Dataset Information:
# Using 'train_df.info()', we present key information about the dataset, including data types, non-null counts, and memory usage. This aids in understanding the overall composition of the data.
train_df.info()

# Handling Missing Values:
# To assess data completeness, we check for missing values in the dataset using 'train_df.isnull().sum()'. Any missing values can have implications for subsequent analysis.
train_df.isnull().sum()

# Missing Values Visualization:
# Visualizing missing values is facilitated by the 'missingno' library. The 'mn.bar()' function is employed to create a bar chart highlighting the presence of missing values in the dataset.
mn.bar(train_df, color='red', figsize=(20, 15))
```

This comprehensive section covers the definition of column names, loading of the dataset, display of dataset information, handling of missing values, and a visualization of missing values. It provides a thorough introduction to the initial steps of data exploration and preprocessing in the context of the project.

** results : **

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/3864fb17-a94c-4aac-aa19-52d46ea133a4)

> The DataFrame is displayed with several columns such as duration, protocol_type, service, flag, src_bytes, dst_bytes, etc. 

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/b86bd8d9-6623-4690-89b5-ab20fcff259c)

>  the info() method output for the train_df DataFrame. This method provides a concise summary of the DataFrame, including the number of non-null entries in each column and the data type of each column. It shows that train_df has 22544 entries with 43 features, and all of the features have non-null integer or floating-point values.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/33cf5be9-89a4-4bd5-b8c9-f2f2887143b7)

>  the result of the isnull().sum() method called on train_df. This method is used to count the number of missing values (NaNs) in each column of the DataFrame. The output indicates that there are no missing values in the DataFrame across all columns.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/12c16e3d-6c16-4009-9cdb-012c1908285a)

> a bar plot created using the pandas plotting interface. Each bar represents a feature in the DataFrame, and the height of the bar seems to correspondto the number of non-null entries for that feature. Since all the bars are of the same height and fill the y-axis scale up to the total number of entries (22544), this corroborates the finding from the isnull().sum() output that there are no missing values in the dataset. The color 'red' and figure size (20, 15) were specified for the visualization.


