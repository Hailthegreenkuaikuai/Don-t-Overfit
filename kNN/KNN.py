import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
import numpy as np

train_data = pd.read_csv('train.csv').drop('id', axis=1)
test_data = pd.read_csv('test.csv').drop('id', axis=1)
X = train_data.drop('target', axis=1)
y = train_data['target']

neigh = KNeighborsClassifier(n_neighbors=17)

sc = StandardScaler()
X = sc.fit_transform(X)
test_data = sc.transform(test_data)

selector = SelectKBest(score_func=f_classif, k=44)
X_selected = selector.fit_transform(X, y)
test_data_selected = selector.transform(test_data)

scores = cross_val_score(neigh, X_selected, y, cv=5)  # 5-fold cross-validation
mean_score = np.mean(scores)
print(f'Cross-Validation Score with k: {mean_score:.4f}')

neigh.fit(X_selected, y)
neigh.kneighbors_graph()

test_predictions = neigh.predict(test_data_selected)

output = pd.DataFrame({'id': range(250, 250 + len(test_predictions)), 'target': test_predictions})
output.to_csv('predictions.csv', index=False)