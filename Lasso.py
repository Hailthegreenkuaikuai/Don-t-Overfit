import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('train.csv').drop('id', axis=1)
X = train_data.drop('target', axis=1)
y = train_data['target']

sk_folds = StratifiedKFold(n_splits=10, shuffle=True)
lasso = Lasso()

pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('lasso', lasso) 
])

param_grid = {
    'alpha': [0.045, 0.05, 0.1],  
    'max_iter' : [30000, 50000, 100000]
}

grid_search = GridSearchCV(estimator=pipeline.named_steps['lasso'], param_grid=param_grid, cv=sk_folds)
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X)
mse = mean_squared_error(y, y_val_pred)
print(f'Mean Squared Error: {mse}')

test_data = pd.read_csv('test.csv').drop('id', axis=1)
test_predictions = best_model.predict(test_data)

test_predictions_binary = [1 if pred >= 0.5 else 0 for pred in test_predictions]

output = pd.DataFrame({'id': range(250, 250 + len(test_predictions)), 'target': test_predictions})
output.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")