import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Veri setini yükleyelim
veri = pd.read_csv('/kaggle/input/housing-price-prediction-data/housing_price_dataset.csv')# Veri setini yükleyelim
print(veri)
print(veri.columns)
veri = veri.drop('Neighborhood',axis=1)
veri.head()
veri['Price'] = round(veri["Price"]).astype('int') #int değere yuvarlama yapıldı-hesaplama kolaylığı
veri.head()
# Bağımlı değişken (y) ve bağımsız değişkenler (X) olarak ayıralım
X = veri[['SquareFeet']]
y = veri['Price']
# Veriyi eğitim ve test setlerine ayırma
X = pd.get_dummies(veri[['SquareFeet']])
y = veri['Price']
# Veriyi eğitim ve test setlerine bölelim (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
}

# Modelleri eğitme ve değerlendirme
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Regresyon metriklerini hesapla
    mse = mean_squared_error(y_test, y_pred)  # Ortalama kare hatası
    r2 = r2_score(y_test, y_pred)  # R² skoru
    print(f"{model_name} - MSE: {mse:.2f}, R²: {r2:.2f}")
