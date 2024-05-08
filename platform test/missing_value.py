import pandas as pd

# Load the data from the CSV file
data = pd.read_csv("E-data_train3.csv")

# Calculate the number of missing values per column
missing_values = data.isnull().sum()

# Calculate the total number of entries in the dataset
total_entries = len(data)

# Calculate the missing rate per column
missing_rate = (missing_values / total_entries) * 100

# Create a DataFrame to display the missing values and missing rate
missing_info = pd.DataFrame(
    {"Missing Values": missing_values, "Missing Rate (%)": missing_rate}
)

# Print the DataFrame in a tabular format
print(missing_info)
