import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    def clean_text(text):
        return text.lower()
    
    df['cleaned_review'] = df['Review'].apply(clean_text)

    X = df['cleaned_review']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    return X_train_vectors, X_test_vectors, y_train, y_test
