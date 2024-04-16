# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:52:15 2024

@author: ZC
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

# Read data from CSV file
df = pd.read_csv("duzenlenmis_otel_veritabani_DB3.csv")

# Remove observations with missing values from the dataset
df.dropna(inplace=True)


# User preferences
kullanici_tercihleri = {'A la Carte Restoran':7,   'Asansör':1, 'Açık Restoran': 1, 'Kapalı Restoran': 1, 'Açık Havuz':5, 'Kapalı Havuz':1,
                        'Bedensel Engelli Odası':1, 'Bar':7, 'Su Kaydırağı': 1, 'Balo Salonu':0, 'Kuaför':1, 'Otopark':1
                        }

# Filter hotels based on user preferences
filtreli_df = df.copy()
for ozellik, deger in kullanici_tercihleri.items():
    filtreli_df = filtreli_df[filtreli_df[ozellik] == deger]

# Determine independent and dependent variables
X = filtreli_df.drop(['otel_ad', 'fiyat', 'il', 'ilce'], axis=1) # Independent variables
y = filtreli_df['fiyat'] # Dependent variable

# If the filtered dataset is empty, print an error message
if len(X) == 0:
    print("No hotel found matching your specified preferences.")
else:
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the price of the hotel based on the user's preferences
    kullanicinin_verisi = []
    for ozellik in X.columns:
        if ozellik in kullanici_tercihleri:
            kullanicinin_verisi.append(kullanici_tercihleri[ozellik])
        else:
            kullanicinin_verisi.append(0)  # If the user didn't specify this feature, default it to 0

    tahmin_edilen_fiyat = model.predict([kullanicinin_verisi])

    print("Predicted Price:", tahmin_edilen_fiyat[0])
