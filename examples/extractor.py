import pandas as pd

data = ''' 
'''

# Remove square brackets and split the data by spaces and create a list of lists
cleaned_data = [row.strip('[]') for row in data.split('\n') if row.strip()]
numbers = [list(map(float, row.split())) for row in cleaned_data]

# Create a DataFrame from the data
df = pd.DataFrame(numbers, columns=["alpha", "beta", "gamma"])

# Export the DataFrame to an Excel file
df.to_excel("output.xlsx", index=False)
