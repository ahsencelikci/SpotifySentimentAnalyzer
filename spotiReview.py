import pandas as pd
import os
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np  # NaN değerlerle çalışmak için numpy kütüphanesini ekleyin


def clean_text(text):
    if pd.isna(text):  # text NaN ise
        return ''  # Boş bir string döndür
    text = re.sub(r'\W', ' ', text)  # Özel karakterleri boşlukla değiştir
    text = re.sub(r'\s+', ' ', text)  # Birden fazla boşluğu tek boşlukla değiştir
    text = text.strip()  # Başındaki ve sonundaki boşlukları sil
    return text

# Dosya yolunu güncelleyin
file_path = 'C:/spotifyReview/DATASET.csv'  
df = pd.read_csv(file_path)  

# Sütunları kontrol et
print("Sütunlar:", df.columns)

# Boş olup olmadığını kontrol et
if df.empty:
    print("Veri seti boş!")
else:
    # Temizleme işlemini gerçekleştir
    df['cleaned_comments'] = df['Review'].apply(clean_text)
    print(df['cleaned_comments'].head())  # Temizlenmiş yorumların ilk beşini yazdır

