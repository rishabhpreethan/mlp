import pandas as pd

# Load the dataset to count the number of houses with 5 or more rooms
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Count the number of houses with 5 or more rooms
houses_with_5_or_more_rooms = dataset[dataset['RM'] >= 5].shape[0]
print(houses_with_5_or_more_rooms)