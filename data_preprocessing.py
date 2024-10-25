import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_and_preprocess_data(file_path):
    # CSV dosyasını oku
    df = pd.read_csv(file_path)

    # Metni temizleme
    def clean_text(text):
        return text.lower()
    
    df['cleaned_review'] = df['Review'].apply(clean_text)

    # Özellikleri ve etiketleri ayırma
    X = df['cleaned_review']
    y = df['label']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # CountVectorizer ile özellik çıkartma
    vectorizer = CountVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    return X_train_vectors, X_test_vectors, y_train, y_test
