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

# Classification of Network Traffic and Distribution of Attack Types

Based on the provided code snippets and results, here is a draft section of a report that discusses the classification of network traffic and the distribution of attack types in the training dataset:

---

# Classification of Network Traffic and Distribution of Attack Types

```python


# Load data from a CSV file into a DataFrame using specified column names
test_df = pd.read_csv("KDDTest+.txt", names=col_names)

# Display the loaded DataFrame
test_df

# Get unique values in the 'class' column of the training DataFrame
train_df['class'].unique()

# Calculate the counts of each class label in the training dataset
classLabel_counts = train_df.groupby(['class']).size()

# Display the class label counts
classLabel_counts

# Calculate the percentage of each class label with respect to the total size of the training dataset
per_classLabels = classLabel_counts / train_df.shape[0] * 100

# Display the percentages of class labels
per_classLabels

# Create a bar plot to visualize the class label counts
fig = plt.figure(figsize=(20, 10))
r_ = [round(each, 2) for each in per_classLabels.values]
ax = fig.add_subplot(111)
ax.bar(per_classLabels.index, per_classLabels.values, color=["mediumaquamarine", 'c', 'darkblue', 'tomato', 'navy'], edgecolor='black')
ax.set_xticklabels(per_classLabels.index, rotation=45)
ax.set_xlabel("Feature Name", fontsize=20)
ax.set_ylabel("Count", fontsize=20)
ax.set_title("Feature 'Class' label counts", fontsize=20)

# Annotate the bar plot with percentage values
for i in range(len(per_classLabels.values)):
    plt.annotate(str(r_[i]), xy=(per_classLabels.index[i], r_[i] + 1), ha='center', va='bottom')

# Define a dictionary to map attack labels to their corresponding categories
attack_mapping = {
    'land': 'Dos', 'neptune': 'Dos', 'smurf': 'Dos', 'pod': 'Dos', 'back': 'Dos', 'teardrop': 'Dos',
    'portsweep': 'Probe', 'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'normal': 'normal'
}

# Define a function to encode the attack labels into attack types using the mapping dictionary
def encode_attack(vec):
    return attack_mapping.get(vec, 'normal')

# Create a new column 'attack_type' based on the encoding of the 'class' column
train_df['attack_type'] = train_df['class'].apply(encode_attack)

# Display the first 10 rows of the DataFrame with the 'attack_type' column
train_df.iloc[:10, -5:]

# Calculate the counts of different attack types in the dataset
train_df.groupby('attack_type').size()

# Calculate the percentage of data held by different attack types
percent_data = (train_df.groupby('attack_type').size()) / train_df.shape[0] * 100

# Display the percentage of data held by different attack types
percent_data

# Create a bar plot to visualize the counts of different attack types
fig = plt.figure(figsize=(10, 8))
r_ = [round(each, 2) for each in percent_data.values]
ax = fig.add_subplot(111)
ax.bar(percent_data.index, percent_data.values, color=['red', 'green', 'orange', 'blue', "mediumaquamarine"], edgecolor='black')
ax.set_xticklabels(percent_data.index, rotation=45)
ax.set_xlabel("Attack type", fontsize=20)
ax.set_ylabel("Count", fontsize=20)
ax.set_title("Attacks type data counts", fontsize=20)

# Annotate the bar plot with percentage values
for i in range(len(percent_data.values)):
    plt.annotate(str(r_[i]), xy=(percent_data.index[i], r_[i] + 1), ha='center', va='bottom')
```

#### Data Acquisition
The testing dataset, referred to as `test_df`, was loaded from a text file named "KDDTest+.txt" using the `pd.read_csv` function from the pandas library. Column names were assigned to the dataset upon import to maintain consistency with the training dataset structure. `test_df` consists of 22544 rows, each row representing a network connection, with 43 features describing each connection.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/ed216812-782f-4350-bc9f-b70ede7ea5e6)


#### Class Label Analysis
The `train_df` dataset was analyzed to determine the unique classes of network traffic, which represent the different types of connections, including various kinds of attacks and normal traffic. Using the `.unique()` method on the 'class' column, a variety of attack categories were identified.

Subsequent to identifying unique classes, we computed the size of each class label in the dataset using `groupby(['class']).size()`. This provided the raw counts of instances for each class label within `train_df`.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/a0826480-5c8c-49f9-bf99-3f5d9c77a103)

#### Distribution of Class Labels
To put these counts into perspective, we calculated the percentage of each class label relative to the total number of instances in `train_df`, which contains 125973 entries. A bar chart was generated to visualize this distribution. The bar chart clearly illustrates that the 'normal' class label constitutes the majority of the instances, followed by various types of network attacks.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/efbba0b6-628e-445d-bff7-f68b14d7e686)


#### Attack Type Categorization
A critical part of the analysis was the categorization of these class labels into broader attack types: Dos

(DoS), Probe, U2R (User to Root), R2L (Remote to Local), and Normal. This was achieved by applying a custom function `encode_attack` that maps each specific attack class to its corresponding attack type. For instance, 'land', 'neptune', 'smurf', 'pod', 'back', and 'teardrop' are all classified as DoS attacks.

The new variable `attack_type` was created in `train_df` to hold these broader categories. The first 10 entries of the updated DataFrame were displayed to verify the successful application of the `encode_attack` function.

#### Proportion of Attack Types
The distribution of the broader attack types within `train_df` was then quantified by grouping the data by `attack_type` and calculating the size of each group. The percentage representation of each attack type was computed relative to the entire dataset. A second bar chart was created to represent this information, which provided a clear visualization of the prevalence of Normal traffic compared to actual attack types, with Normal traffic significantly outnumbering the attack instances.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/be99b086-d05b-4e4c-b6b2-36ba70d21226)


#### Visualization


These visualizations serve as an integral part of exploratory data analysis, providing immediate insights into the balance of the dataset and the relative frequency of different types of network activities captured in the data. The analysis indicates that while the dataset contains a diverse set of attack classifications, Normal traffic remains the most prevalent class, accounting for a significant portion of the network data.

This information is crucial for subsequent stages of data preprocessing, feature engineering, and the development of machine learning models, as it highlights the need for strategies to address class imbalance and ensure robust model performance across all categories of network traffic.



## Dependency of "flags" in attack type:

```python
# Create a new figure for plotting with a specified size
fig = plt.figure(figsize=(10, 8))

# Calculate a cross-tabulation (crosstab) of 'flag' and 'attack_type' columns in the training DataFrame
avg_pro = pd.crosstab(train_df['flag'], train_df['attack_type'])

# Normalize the crosstab by dividing each row by the sum of that row, converting it to a percentage
avg_pro.div(avg_pro.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['indigo', 'gold', 'teal', 'olive', 'slategrey'])

# Add a title to the plot
plt.title('Dependency of "flags" in attack type', fontsize=20)

# Label the x-axis
plt.xlabel('Flags', fontsize=20)

# Add a legend to the plot to distinguish attack types
plt.legend()

# Display the plot
plt.show()
```
This code creates a bar plot to visualize the dependency of different flags on different attack types by performing a cross-tabulation and normalizing the results to show the distribution of flags within each attack type.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/53ae57c0-3595-4a80-97e9-3c38fc7c9951)


   - This chart displays the proportion of each attack type for different "flags" values. Flags in network traffic often indicate the type of connection or session state (such as established, reset, or synchronized).
   - Each color in the bars represents a different attack type (DoS, Probe, U2R, R2L, Normal).
   - The y-axis represents the proportion (likely normalized to a range between 0 and 1) of attack types within each flag category.
   - The chart shows that some flags are more associated with certain types of attacks. For instance, one specific flag may have a higher proportion of DoS attacks, whereas another might be more associated with Normal traffic.

## Dependency of "Protocols" in Attack types:

```python
# Create a new figure for plotting with a specified size
fig = plt.figure(figsize=(10, 8))

# Calculate a cross-tabulation (crosstab) of 'protocol_type' and 'attack_type' columns in the training DataFrame
avg_pro = pd.crosstab(train_df['protocol_type'], train_df['attack_type'])

# Normalize the crosstab by dividing each row by the sum of that row, converting it to a percentage
avg_pro.div(avg_pro.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['indigo', 'gold', 'teal', 'olive', 'slategrey'])

# Add a title to the plot
plt.title('Dependency of "Protocols" in Attack types', fontsize=20)

# Label the x-axis
plt.xlabel('Protocol Types', fontsize=20)

# Add a legend to the plot to distinguish attack types
plt.legend()

# Display the plot
plt.show()
```
This code creates a bar plot to visualize the dependency of different protocols on different attack types by performing a cross-tabulation and normalizing the results to show the distribution of protocols within each attack type.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/7a04fe01-b791-40e4-b2b0-76f8e6866db6)


   - This chart shows the distribution of attack types over different network protocols such as TCP, UDP, ICMP, etc.
   - Like the flags chart, different colors represent different attack types.
   - The y-axis again seems to represent the proportion of each attack type within the given protocol type.
   - This kind of analysis is important to understand which protocols are more vulnerable to specific types of attacks.

### Analysis:
- The charts are useful for identifying patterns in the data, such as if certain flags or protocols are more prone to specific attack types. 
- For example, if a flag has a very high proportion of DoS attacks, it might indicate that sessions with this flag should be monitored more closely for potential DoS activity.
- Similarly, if a

protocol is predominantly associated with normal activity, this might suggest that traffic using this protocol is less likely to be malicious. Conversely, if a protocol is mostly associated with attack traffic, it could be a vector that security measures need to address more rigorously.
- These visualizations could inform security professionals about the typical profiles of network traffic that are associated with different types of attacks, aiding in the development of targeted security measures.
- However, it's important to note that correlation does not imply causation. While these visualizations show a relationship between flags or protocols and attack types, they do not necessarily indicate that one causes the other. Further statistical analysis would be required to determine if these relationships are significant and to what extent they might influence the likelihood of an attack.

In summary, these visualizations are part of exploratory data analysis (EDA) that can provide insights into the structure and patterns within network traffic data. Such analysis is critical in the field of cybersecurity, where understanding the characteristics of

traffic can lead to better anomaly detection and intrusion prevention strategies. The charts could also highlight areas where data collection might need to be more robust, or where further investigation could be beneficial for a more comprehensive understanding of network vulnerabilities. 

## Dependency of services

```python
# Create a new figure for plotting with a specified size
fig = plt.figure(figsize=(10, 8))

# Calculate a cross-tabulation (crosstab) of 'service' and 'attack_type' columns in the training DataFrame
avg_pro = pd.crosstab(train_df['service'], train_df['attack_type'])

# Normalize the crosstab by dividing each row by the sum of that row, converting it to a percentage
avg_pro.div(avg_pro.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['indigo', 'gold', 'teal', 'olive', 'slategrey'])

# Add a title to the plot
plt.title('Dependency of "Services" in Attack types', fontsize=20)

# Label the x-axis
plt.xlabel('Service Types', fontsize=20)

# Add a legend to the plot to distinguish attack types
plt.legend()

# Display the plot
plt.show()
```

This code creates a bar plot to visualize the dependency of different service types on different attack types by performing a cross-tabulation and normalizing the results to show the distribution of services within each attack type.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/9d449f69-0b1a-44cb-a4f1-ad2f078104fc)


Here's an analysis of the chart:

- The x-axis represents different network services, which are likely to be from a dataset that includes network traffic data. Each bar represents a unique service type, such as HTTP, FTP, SMTP, etc.
- The y-axis indicates the proportion, which ranges from 0 to 1, signifying the percentage contribution of each attack type to the service on the x-axis.
- The colors in the bars represent different attack types: DoS (Denial of Service), Probe, U2R (User to Root), and normal traffic. Each color segment's length within a bar represents the proportion of that attack type for the corresponding service.
- The chart shows that some services are predominantly associated with a single type of attack (e.g., a service might show a large purple segment, indicating a high proportion of DoS attacks).

This kind of visualization is particularly useful in cybersecurity for identifying which network services are most associated with certain types of attacks. Security teams can use this information to prioritize monitoring and protective measures for services that are frequently targeted.

However, this chart appears to be quite dense with many services listed on the x-axis, which makes it difficult to read specific service names or to discern the distribution clearly for each service. This suggests that a different type of visualization or a filtered view showing only the most common

services might be more effective for detailed analysis. For example, a series of pie charts for the most frequently attacked services or a heat map could provide clearer insights.

## Encoding attaques :


```python
# Define a dictionary to encode attack types into numerical values
attack_encoding = {'normal': 0, 'Dos': 1, 'Probe': 2, 'R2L': 3}

# Define a function to apply the attack encoding to the 'attack_type' column and create a new 'intrusion_code' column
def attack_encode(value):
    return attack_encoding.get(value, 4)  # Default to 4 if value is not found in the dictionary

# Apply the attack_encode function to the 'attack_type' column and create a new 'intrusion_code' column
train_df['intrusion_code'] = train_df['attack_type'].apply(attack_encode)

# Display the first 10 rows of the DataFrame with the new 'intrusion_code' column
train_df.iloc[:10, -5:]

# Filter the DataFrame to show the first 10 rows where 'intrusion_code' is equal to 2
train_df[train_df['intrusion_code'] == 2].iloc[:10, -5:].head()

# Copy the original DataFrames for backup
train_df_back = train_df
test_df_back = test_df

# Drop columns 'class', 'difficulty level', and 'attack_type' from both train and test DataFrames
train_df = train_df.drop(columns=['class', 'difficulty level', 'attack_type'])
test_df = test_df.drop(columns=['class', 'difficulty level', 'attack_type'])

# Perform one-hot encoding on both the train and test DataFrames to convert categorical variables into numerical form
train_df_new = pd.get_dummies(train_df)
test_df_new = pd.get_dummies(test_df)

# Calculate the correlation between features and the target variable 'intrusion_code'
highly_correlated = train_df_new.corr().abs()['intrusion_code'].sort_values(ascending=False)

# Print the top 30 highly correlated features with respect to the target variable 'intrusion_code'
print(highly_correlated[:30])

# Select only the top 30 features from both the train and test DataFrames for modeling
train_df_new = train_df_new[list(highly_correlated[:30].index)]
test_df_new = train_df_new[list(highly_correlated[:30].index)]
```

Explanation:
1. A dictionary `attack_encoding` is defined to map attack types to numerical values.
2. A function `attack_encode` is defined to apply this encoding to the 'attack_type' column, creating a new 'intrusion_code' column in the DataFrame.
3. The first 10 rows of the DataFrame with the new 'intrusion_code' column are displayed.
4. Rows where 'intrusion_code' is equal to 2 are filtered and displayed.
5. Original DataFrames are backed up as `train_df_back` and `test_df_back`.
6. Columns 'class', 'difficulty level', and 'attack_type' are dropped from both train and test DataFrames.
7. One-hot encoding is applied to convert categorical variables into numerical form for both train and test DataFrames.
8. The correlation between features and the target variable 'intrusion_code' is calculated.
9. The top 30 highly correlated features are printed.
10. Only the top 30 features are selected from both the train and test DataFrames for modeling.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/1c485648-75ab-40ed-abef-2d72309f315c)

The image shows the first lines of the DataFrame after applying this function. We can see that 'normal' type attacks have an intrusion code of 0, and the 'neptune' attacks, which are 'Dos' type attacks, have an intrusion code of 1. This transforms the categorical data into a numeric format that can be more easily used in machine learning models.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/efdda8f1-ebc3-4912-b767-4967405e972d)

The image shows a snippet of this filtered subset. We can observe different types of probe attacks ('ipsweep', 'portsweep', 'nmap', 'satan') and their corresponding destination host service error rate ('dst_host_srv_rerror_rate'). The values of 'difficulty level' vary, which could indicate the difficulty in detecting or responding to these attacks. Each line representing a 'Probe' type attack has been assigned an intrusion code of 2, in accordance with the previously established coding logic.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/a25d90ee-4db0-42e2-bd93-3e290985fe1e)

In the table, the selected columns display various statistics of network connections, the original attack type ('class'), difficulty level ('difficulty level'), categorized attack type ('attack_type'), and the corresponding intrusion code ('intrusion_code'). There are instances of 'normal' traffic and 'Dos' type attacks, mainly 'neptune'. The data reflect the outcome of the mapping performed, with 'neptune' attacks correctly identified as 'Dos' and coded as 1.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/aced4087-e694-43c6-b464-f8e914248bde)

The table shows an excerpt from the DataFrame test_df after applying a coding function to the 'attack_type' column, which was added using the names of attacks in the 'class' column. The attack_encode function converted the names of the attacks into a numerical code stored in the new 'intrusion_code' column. In this excerpt, 'neptune' attacks are categorized as 'Dos' with an intrusion code of 1, while 'normal', 'saint', and 'mscan' are considered normal traffic with an intrusion code of 0. This demonstrates that the function has been correctly applied to reflect the type of attack in a numerical form, which is useful for subsequent analysis steps.

![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/2d253c3b-709a-4ff2-9176-f0c34918b8ea)



![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/b1566f40-8377-42a0-a715-d9e22b063816)


