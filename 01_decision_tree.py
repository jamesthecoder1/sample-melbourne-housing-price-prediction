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

# import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# define and fit the model
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

# import mean_absolute_error to evaluate the model
from sklearn.metrics import mean_absolute_error

# get predicted prices on validation data
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
