import pandas as pd
import os

#  Directory containing the files
# directory = 'datasets/timeseries/nasdaq/nasdaq'

# # List to hold dataframes
# dataframes = []

# # Loop through each file in the directory
# for year in range(2008, 2024):
#     file_path = f'datasets/timeseries/nasdaq/nasdaq/{year}_Global_Markets_Data.csv'
#     # Read the CSV file
#     df = pd.read_csv(file_path)
#     # Append the dataframe to the list
#     dataframes.append(df)

# # Concatenate all dataframes
# combined_df = pd.concat(dataframes, ignore_index=True)

# # Save the combined dataframe to a new CSV file
# combined_df.to_csv('datasets/timeseries/nasdaq/nasdaq/Combined_Global_Markets_Data.csv', index=False)

# print("Combined file created successfully.")


file_path = f'datasets/timeseries/nasdaq/nasdaq/2008_Global_Markets_Data.csv'
df = pd.read_csv(file_path)
df_head = df.head()
df_head.to_csv('datasets/timeseries/nasdaq/nasdaq/2008_Global_Markets_Data_head.csv')
df_tail = df.tail()
df_tail.to_csv('datasets/timeseries/nasdaq/nasdaq/2008_Global_Markets_Data_tail.csv')

file_path = f'datasets/timeseries/nasdaq/nasdaq/2010_Global_Markets_Data.csv'
df = pd.read_csv(file_path)
df_head = df.head()
df_head.to_csv('datasets/timeseries/nasdaq/nasdaq/2010_Global_Markets_Data_head.csv')
df_tail = df.tail()
df_tail.to_csv('datasets/timeseries/nasdaq/nasdaq/2010_Global_Markets_Data_tail.csv')
