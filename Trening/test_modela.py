import joblib


vectorizer = joblib.load('model/vectorizer.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
model = joblib.load('model/logreg_model.pkl')  


def preprocess(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    text = ' '.join(text.split())
    return text


def predict_category(product_title):
    cleaned = preprocess(product_title)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)
    category = label_encoder.inverse_transform(pred)
    return category[0]


if __name__ == "__main__":
    print("Unesi naziv proizvoda za predikciju kategorije (ili 'exit' za izlaz):\n")
    while True:
        title = input("Proizvod: ")
        if title.lower() == 'exit':
            print("Zatvaram program.")
            break
        category = predict_category(title)
        print(f"Predviđena kategorija: {category}\n")
