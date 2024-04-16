# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:15:12 2024

@author: ZC
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv("duzenlenmis_otel_veritabani_DB3.csv")
data.dropna(inplace=True)

# Get the name of the hotel the user previously visited
previous_hotel_name = "Adam Eve"  # For example

# Find the features of the previous hotel
previous_hotel = data[data['hotel_name'] == previous_hotel_name].drop(columns=['hotel_name', 'price', 'city', 'district'])

# Get the features of other hotels in the dataset
other_hotels = data.drop(columns=['hotel_name', 'price', 'city', 'district'])

# Calculate similarity scores using cosine similarity
similarity_scores = cosine_similarity(previous_hotel, other_hotels)

# Find indices of the most similar hotels (excluding itself)
most_similar_hotel_indices = similarity_scores.flatten().argsort()[-4:-1]

# Find the names of the recommended hotels
recommended_hotel_names = data.loc[most_similar_hotel_indices, 'hotel_name']

print("Recommended hotels: ")
for hotel_name in recommended_hotel_names:
    print(hotel_name)
