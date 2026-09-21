# Melbourne Housing Price Prediction

Predicts home prices in the Melbourne housing market using Decision Tree and Random Forest regression models, built step-by-step to demonstrate model development and improvement.

## What This Project Does
- Loads and cleans the Melbourne housing dataset
- Splits data into training and validation sets to avoid overfitting
- Trains and evaluates a baseline Decision Tree model
- Tunes the Decision Tree to find the optimal tree size
- Builds a Random Forest model for improved accuracy
- Compares all three approaches using Mean Absolute Error (MAE)

## Skills Demonstrated
- Data cleaning with pandas (handling missing values)
- Train/validation splitting to avoid in-sample evaluation
- Model tuning (finding optimal `max_leaf_nodes`)
- Model evaluation using MAE
- Comparing model performance across different approaches

## Project Structure
- `01_decision_tree.py` — baseline model using a single Decision Tree
- `02_decision_tree_tuning.py` — tuning tree depth to reduce overfitting/underfitting
- `03_random_forest.py` — final model using Random Forest for improved accuracy

## How to Run
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run each script individually, in order:
python 01_decision_tree.py
python 02_decision_tree_tuning.py
python 03_random_forest.py

## Results Comparison
| Model | Validation MAE |
|---|---|
| Decision Tree (baseline) | 251876.65 |
| Decision Tree (tuned) | [fill in] |
| Random Forest | [fill in] |

## Dataset
Melbourne Housing Market dataset (`melb_data.csv`), located in `data-sample/`