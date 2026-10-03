from sklearn.feature_extraction import DictVectorizer

# Sample list of dictionaries
data = [{'city': 'London', 'temperature': 12}, {'city': 'Dubai', 'temperature': 33}]

# Initialize vectorizer (setting sparse=False for a NumPy array)
vec = DictVectorizer(sparse=False)

# Fit and transform the data
X = vec.fit_transform(data)
print(X)