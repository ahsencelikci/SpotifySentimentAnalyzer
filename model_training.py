import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Veri setini yükle
file_path = 'C:/spotifyReview/DATASET.csv'  # Dosya yolunu güncelleyin
df = pd.read_csv(file_path)

# NaN değerleri kontrol et
print("NaN değerler:", df.isnull().sum())

# NaN değerleri içeren satırları sil
df = df.dropna()

# Gerekli sütunları seçin ve veriyi ayırın
X = df['Review']
y = df['label']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Metin verilerini vektörleştirme
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Modeli tanımla
model = MultinomialNB()

# Modeli eğit
model.fit(X_train_vectors, y_train)

# Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test_vectors)

# Sonuçları değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy:.2f}")

# Daha ayrıntılı bir rapor al
print(classification_report(y_test, y_pred))

# Modeli kaydet
joblib.dump(model, 'spotify_review_model.pkl')

# Yeni bir yorum üzerinde tahmin yap
new_review = ["I love the playlists on Spotify!"]
new_review_vector = vectorizer.transform(new_review)
prediction = model.predict(new_review_vector)

print(f"Yeni yorum tahmini: {prediction[0]}")
