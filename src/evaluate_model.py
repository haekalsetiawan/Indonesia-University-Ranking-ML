from sklearn.metrics import accuracy_score

def evaluate_classification_model(model, X_test, y_test):
    # Prediksi menggunakan model
    y_pred = model.predict(X_test)

    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
