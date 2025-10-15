# product-category-classifier
Ovaj projekt je jednostavan model za automatsko klasifikovanje proizvoda u odgovarajuće kategorije na osnovu **naslova proizvoda**. 

Koristi se **scikit-learn** pipeline sa TF-IDF vektorizacijom i klasifikacijom pomoću modela poput `Logistic Regression`, `Random Forest`, i `Naive Bayes`.

## Struktura foldera

<pre lang="nohighlight"> ```plaintext product-category-classifier/ ├── Treniranje modela/ │ ├── train_model.py │ ├── test_modela.py │ ├── products.csv │ └── model/ │ ├── logreg_model.pkl │ ├── vectorizer.pkl │ └── label_encoder.pkl ├── notebooks/ │ └── data_exploration_and_cleaning_and_model_training.ipynb ├── data/ │ └── products.csv ├── .gitignore └── README.md ``` </pre>
## Dataset

Koristi se CSV fajl `products.csv` koji sadrži kolone:

- `Product Title` — naslov proizvoda
- `Category Label` — stvarna kategorija
- `Number_of_Views`, `Merchant Rating`, `_Product Code` itd.

- Treniraj model tako što pokreneš skriptu:

```bash
python "Trening/train_model.py"

Primjer koriscenja:
Proizvod: samsung galaxy s21 ultra
Predviđena kategorija: Mobile Phones
