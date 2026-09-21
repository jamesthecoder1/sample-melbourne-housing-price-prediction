# import pandas to handle data
import pandas as pd

# set data filepath and read data into a DataFrame
melb_filepath = "data-sample/melb_data.csv"
melb_data = pd.read_csv(melb_filepath)

# filter data to remove missing values
filtered_melb_data = melb_data.dropna(axis=0)

# split data into features and target
y = filtered_melb_data.Price
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melb_data[melb_features]

# import RandomForestRegressor, MAE, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

forestModel = RandomForestRegressor(random_state=1)
forestModel.fit(train_X, train_y)
melb_predictions = forestModel.predict(val_X)
print(mean_absolute_error(val_y, melb_predictions))