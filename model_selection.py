from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# generate sample data
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)

# select model based on the size of the data
if X.shape[0] < 1000:
    model = LogisticRegression()
else:
    model = RandomForestClassifier()

# train the model
model.fit(X, y)

# predict
predictions = model.predict(X)
print("predictions:", predictions[:5])