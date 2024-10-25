import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib


file_path = 'C:/spotifyReview/DATASET.csv' 
df = pd.read_csv(file_path)


print("NaN değerler:", df.isnull().sum())


df = df.dropna()


X = df['Review']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train_vectors, y_train)

y_pred = model.predict(X_test_vectors)

accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy:.2f}")


print(classification_report(y_test, y_pred))

joblib.dump(model, 'spotify_review_model.pkl')

new_review = ["I love the playlists on Spotify!"]
new_review_vector = vectorizer.transform(new_review)
prediction = model.predict(new_review_vector)

print(f"Yeni yorum tahmini: {prediction[0]}")
