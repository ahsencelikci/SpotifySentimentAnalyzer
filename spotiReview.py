import pandas as pd
import os
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np  


def clean_text(text):
    if pd.isna(text):  
        return '' 
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip()  
    return text


file_path = 'C:/spotifyReview/DATASET.csv'  
df = pd.read_csv(file_path)  


print("Sütunlar:", df.columns)


if df.empty:
    print("Veri seti boş!")
else:
 
    df['cleaned_comments'] = df['Review'].apply(clean_text)
    print(df['cleaned_comments'].head())

