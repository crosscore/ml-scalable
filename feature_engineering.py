from sklearn.feature_extraction import FeatureHasher
from sklearn.datasets import make_classification
import pandas as pd

# sample data (categorical features)
data = {'feature': ['cat', 'dog', 'bird', 'dog', 'bird'] * 10000}
df = pd.DataFrame(data)

# apply hashing trick
hasher = FeatureHasher(n_features=10, input_type='string')
hashed_features = hasher.fit_transform(df['feature'].apply(lambda x: [x]))
print("hashed features:", hashed_features.toarray()[:5])