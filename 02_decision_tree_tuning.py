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

# import DTR, MAE, and train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# define a function to get MAE for a given max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# for-loop to compare MAE with different values of max_leaf_nodes
candidate_max_leaf_nodes = [5, 50, 500, 5000]
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f"Max leaf nodes: {max_leaf_nodes}  \t\t Mean Absolute Error: {my_mae:.0f}")

# 500 is the best value for max_leaf_nodes because it has the lowest MAE
best_tree_size = 500

# fit the model with best_tree_size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)