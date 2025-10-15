import pandas as pd
import numpy as np
import re
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import os



pd.set_option('display.max_columns', None)

df = pd.read_csv('products.csv')

df.columns = df.columns.str.strip()

df_clean = df.dropna(subset=['Product Title', 'Category Label']).copy()


df_clean['Product Title'] = df_clean['Product Title'].str.lower()

df_clean['Product Title'] = df_clean['Product Title'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', str(x)))

df_clean['Product Title'] = df_clean['Product Title'].str.replace(r'\s+', ' ', regex=True).str.strip()

df_clean['Category Label'] = df_clean['Category Label'].str.strip()

df = df.dropna(subset=['Product Title'])

df['title_length'] = df['Product Title'].apply(len)
df['title_word_count'] = df['Product Title'].apply(lambda x: len(x.split()))
df['title_contains_digit'] = df['Product Title'].apply(lambda x: int(any(char.isdigit() for char in x)))
df['title_digit_count'] = df['Product Title'].apply(lambda x: sum(char.isdigit() for char in x))


#print(df[['Product Title', 'title_length', 'title_word_count', 'title_contains_digit', 'title_digit_count']].head())

df['Number_of_Views'] = df['Number_of_Views'].fillna(df['Number_of_Views'].mean())
df['Merchant Rating'] = df['Merchant Rating'].fillna(df['Merchant Rating'].mean())
df['_Product Code'] = df['_Product Code'].fillna('unknown')
df['Product Title'] = df['Product Title'].fillna('unknown')
df['Category Label'] = df['Category Label'].fillna('Other')

y = df['Category Label']


le = LabelEncoder()
y = le.fit_transform(y)

X = df['Product Title'].fillna('')  


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial NB": MultinomialNB(),
    "Linear SVC": LinearSVC(),
}

for name, model in models.items():
    #print(f"\nModel: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #print(classification_report(y_test, y_pred, target_names=le.classes_))


new_product = ["samsung galaxy s21 ultra 5g 128gb phantom black"]
new_product_vectorized = vectorizer.transform(new_product)

for name, model in models.items():
    pred = model.predict(new_product_vectorized)
    category_name = le.inverse_transform(pred)
    print(f"{name}: {category_name[0]}")


if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(vectorizer, 'model/vectorizer.pkl')

joblib.dump(le, 'model/label_encoder.pkl')

logreg = models["Logistic Regression"]
logreg.fit(X_train, y_train)
joblib.dump(logreg, 'model/logreg_model.pkl')

print("Snimanje zavrseno.")
