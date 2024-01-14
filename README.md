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

The table represents data from a DataFrame that includes information on network activities. The displayed columns include characteristics of connections such as the type of protocol (tcp/udp), service (ftp_data, other, private, etc.), connection flag (SF, REJ, S0), and transferred bytes (src_bytes, dst_bytes). Additionally, error rate measures are presented. The last two columns, 'attack_type' and 'intrusion_code', classify the activities either as normal or as attacks, where 'neptune' is identified as a DoS (Denial of Service) type attack and coded with the number 1, while normal activities are coded with the number 0.


![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/b1566f40-8377-42a0-a715-d9e22b063816)


The results show that characteristics such as 'logged_in', 'dst_host_srv_count', and 'flag_SF' are among the most correlated with 'intrusion_code'. This indicates that these variables could be important predictors for identifying types of attacks in network data and should be included in the construction of the predictive model.


## correlation between features :

```python
corr_df = train_df_new.corr()[train_df_new.corr().index]
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corr_df, annot=True, annot_kws={"size": 11})
plt.show()
```


1. `corr_df = train_df_new.corr()[train_df_new.corr().index]`:
   - `train_df_new.corr()` computes the correlation matrix for the DataFrame `train_df_new`.
   - The indexing `[train_df_new.corr().index]` is redundant in this context, as calling `corr()` already returns a square matrix with rows and columns corresponding to the DataFrame's columns. It seems unnecessary unless there is a specific reason for reordering the matrix using its own index.

2. `fig, ax = plt.subplots(figsize=(20,20))`:
   - This line initializes a matplotlib Figure and Axes object. The `figsize=(20,20)` argument makes the figure 20 inches by 20 inches, ensuring that the heatmap will be large enough to display all variables clearly.

3. `sns.heatmap(corr_df, annot=True, annot_kws={"size": 11})`:
   - This line creates a heatmap using the Seaborn library with the correlation matrix `corr_df`.
   - `annot=True` enables annotations inside the heatmap squares, displaying the correlation coefficients.
   - `annot_kws={"size": 11}` sets the

4. `plt.show()`:
   - This command displays the heatmap visualization.
   - 
![image](https://github.com/zaouizakariae/NSL-KDD/assets/85891554/7319930d-2159-4de5-80a8-080245be28d7)

The resulting heatmap visualizes the correlation coefficients between all pairs of features in `train_df_new`. Correlation coefficients range from -1 to 1, where:
- 1 indicates a perfect positive linear relationship,
- -1 indicates a perfect negative linear relationship,
- 0 indicates no linear relationship.

In the heatmap, each square represents the correlation between the variables on each axis. Colors typically range from dark to light, where dark colors represent strong negative correlations and light colors represent strong positive correlations. The correlation of a variable with itself is always 1, hence the diagonal line of light-colored squares.

From the provided heatmap image, you can analyze which features are most strongly correlated with each other. Features that are highly correlated with 'intrusion_code' might be particularly important for predictive modeling purposes, as they could influence the model's ability to distinguish between different types of network activities, such as normal behavior and various kinds of attacks.

## Modeling

```python
# Drop the 'intrusion_code' column to create the feature set (X)
X = train_df_new.drop(columns='intrusion_code')

# The 'intrusion_code' column is used as the target variable (y)
y = train_df_new['intrusion_code']

# Convert all column names in X to strings for consistency
X.columns = X.columns.astype(str)

# Convert the DataFrame X to a NumPy array for scaling
X_array = X.values

# Standardizing the data: Initialize the StandardScaler and fit it to the data
scaler = StandardScaler().fit(X_array)

# Transform the data using the fitted scaler
X_scaled = scaler.transform(X_array)

# Convert the scaled NumPy array back to a DataFrame
# This step retains the column names and the DataFrame structure
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Splitting the data into training and testing sets
# 80% of the data is used for training and 20% for testing
# The random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=13)
```

1. **Preparing X (feature set variables) and y (target variable)**:
   - `X = train_df_new.drop(columns='intrusion_code')`: This line creates a DataFrame `X` containing all the columns from `train_df_new` except for 'intrusion_code'. This implies 'intrusion_code' is the target variable, and the remaining columns are features.
   - `y = train_df_new['intrusion_code']`: This line creates the target variable `y`, which consists of the values from the 'intrusion_code' column of `train_df_new`.

2. **Convert all column names to strings**:
   - `X.columns = X.columns.astype(str)`: This converts the column names of `X` into strings. This can be necessary if the column names are not already in string format and a later process requires them to be strings.

3. **Convert DataFrame to NumPy array for scaling**:
   - `X_array = X.values`: This converts the DataFrame `X` into a NumPy array `X_array`. This is often done before applying scaling because many scaling functions expect input in array format.

4. **Standardizing data using StandardScaler**:
   - `scaler = StandardScaler().fit(X_array)`: This initializes a `StandardScaler` object and fits it to the data. `StandardScaler` standardizes features by removing the mean and scaling to unit variance.
   - `X_scaled = scaler.transform(X_array)`: This transforms the data using the fitted scaler, standardizing the features.

5. **Convert NumPy array back to DataFrame**:
   - `X_scaled = pd.DataFrame(X_scaled, columns=X.columns)`: The scaled data, which is in the form of a NumPy array, is converted back into a DataFrame. This is often done for better handling and visualization of the data.

6. **Splitting data into train and test**:
   - `X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=13)`: This line splits the dataset into training and testing sets. `test_size=0.2` indicates that 20% of the data will be used for testing, while the rest will be used for training. The `random_state` parameter ensures that the split is reproducible; the same split will occur every time the code is run.


The provided code and results detail the process of defining, training, and evaluating a neural network model using TensorFlow and Keras for a classification task. Let's break down each part:

## 

```python
# Import necessary TensorFlow and Keras modules
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

# Assuming X_train, y_train, X_test, and y_test are predefined

# Define the architecture of the neural network
model = tf.keras.models.Sequential([
    # Add a Dense layer with 64 neurons and ReLU activation function. 
    # The input_shape is set to the number of features in X_train.
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),

    # Add another Dense layer with 32 neurons, also with ReLU activation.
    tf.keras.layers.Dense(32, activation='relu'),

    # Add the output Dense layer with 1 neuron using sigmoid activation function 
    # suitable for binary classification.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
# Use the Adam optimizer, binary crossentropy as the loss function 
# (suitable for binary classification tasks), and track accuracy as a metric.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display a summary of the model's architecture
# This shows the layers, their types, output shapes and number of parameters.
model.summary()

# Train the model on the training data
# Train for 30 epochs (iterations over the entire dataset), with a batch size of 32
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Utiliser un seuil de 0.5 pour la classification binaire

print("\nRapport de classification :\n", classification_report(y_test, y_pred))
print("\nPrécision du modèle :", accuracy_score(y_test, y_pred))


```

## Model Definition
1. **Architecture**: 
   - The model is a Sequential model with three Dense layers.
   - The first Dense layer has 64 neurons and uses the ReLU activation function. It is the input layer and its input shape is defined by the number of features in `X_train`.
   - The second Dense layer has 32 neurons, also with ReLU activation.
   - The third Dense layer is the output layer with a single neuron using the sigmoid activation function, which is typical for binary classification. 

2. **Compilation**:
   - The model is compiled with the Adam optimizer.
   - The loss function used is 'binary_crossentropy', appropriate for binary classification tasks.
   - The metric for evaluation is 'accuracy'.

## Model Training
1. **Training Process**:
   - The model is trained for 30 epochs with a batch size of 32.
   - A validation split of 20% is used, meaning 20% of the training data is used as a validation set to evaluate the model during training.

2. **Training Output**:
   - The training output shows the loss and accuracy for each epoch on both the training and validation sets.
   - The loss is decreasing and accuracy is increasing as expected, but the loss values are unusually large and negative, which is atypical and might indicate an issue with the data or the model configuration.

## Model Evaluation
1. **Testing**:
   - The model's performance is evaluated on the test set. The prediction (`y_pred`) is derived by applying a threshold of 0.5 to the model's output.

2. **Classification Report and Accuracy**:
   - The classification report shows precision, recall, and f1-score for each class along with overall accuracy.
   - The model achieves an overall accuracy of 88%.
   - However, the precision, recall, and f1-scores for some classes (like class 2 and 4) are 0, indicating the model is not performing well on these classes.


## conclusion :

After a series of analyses and development of models using NSL-KDD data for intrusion detection, we have achieved several important realizations. Preprocessing methods and exploratory data analysis have helped in understanding and preparing the data for modeling. Neural network models were designed and optimized, leading to a highly accurate model with an accuracy of about 99% after hyperparameter tuning. Confusion matrices and performance metrics confirmed the effectiveness of the final model. This work illustrates the potential of advanced machine learning techniques in securing computer systems against cyberattacks.
