import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load datasets
restDatasets = pd.read_csv('GoogleReview_Penang.csv')
menuDatasets = pd.read_csv('Restaurants_Penang_Menu.csv')

# Combine restaurant name and cuisine into one column
restDatasets['NameCuisine'] = restDatasets['Restaurant'] + \
    ' ' + restDatasets['Cuisine']

# create tfidf vectorizer
tfidf = TfidfVectorizer(stop_words=None)

# fit and transform the namecuisine column
tfidf_matrix = tfidf.fit_transform(restDatasets['NameCuisine'])

# measure cosine similarity between vectors
similarity = cosine_similarity(tfidf_matrix)

# replace empty values in itemBestSeller to No, itemDescription to be same with itemName, itemAllergens to be None
menuDatasets['itemBestSeller'] = menuDatasets['itemBestSeller'].fillna('No')
menuDatasets['itemDescription'] = menuDatasets['itemDescription'].fillna(
    menuDatasets['itemName'])
menuDatasets['itemAllergens'] = menuDatasets['itemAllergens'].fillna('None')

# combine item name and description together
menuDatasets['itemNameDesc'] = menuDatasets['itemName'] + \
    ' ' + menuDatasets['itemDescription']

# create tfidf vectorizer on menu items
tfidf_menu = TfidfVectorizer(stop_words='english')

tfidf_matrix_menu = tfidf_menu.fit_transform(menuDatasets['itemNameDesc'])

similarity_menu = cosine_similarity(tfidf_matrix_menu)

# start recommendation on restaurants


def filter_restaurants_by_halal(restaurant_data, user_halal):
    if user_halal.lower() == 'yes':
        return restaurant_data[restaurant_data['isHalal'].str.lower() == 'yes']
    return restaurant_data


def normalize_features(feature_array):
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature_array.reshape(-1, 1)).flatten()


def build_feature_matrix(restaurant_data, user_input):
    # Create TF-IDF matrix based on 'NameCuisine'
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        restaurant_data['NameCuisine'])

    # Create aggregated cuisine preference score
    user_cuisines = [cuisine.strip().lower()
                     for cuisine in user_input.split(',')]

    # Initialize cuisine scores to zero
    cuisine_scores = np.zeros(len(restaurant_data))

    # Aggregate cuisine scores based on user preferences
    for cuisine in user_cuisines:
        cuisine_scores += restaurant_data['NameCuisine'].apply(
            lambda x: 1 if cuisine in x.lower() else 0
        ).values

    # Normalize cuisine scores to avoid overweighting
    normalized_cuisine_scores = normalize_features(cuisine_scores)

    # Normalize ratings
    normalized_ratings = normalize_features(restaurant_data['Rating'].values)

    # Combine all features
    combined_matrix = np.hstack([
        tfidf_matrix.toarray(),
        # Add single cuisine score feature
        normalized_cuisine_scores.reshape(-1, 1),
        normalized_ratings.reshape(-1, 1)
    ])

    # Return combined matrix and vectorizer
    return combined_matrix, user_cuisines, tfidf_vectorizer


allergen_diet_similarity = {
    'dairy': 'dairy-free',
    'wheat': 'gluten-free',
}


def normalize_keywords(keywords):
    # Replace synonyms with standard terms
    return [allergen_diet_similarity.get(keyword.strip().lower(), keyword.strip().lower()) for keyword in keywords]


def get_top_menu_items(restaurantName, menu_data, user_allergy, user_diet, max_items=3):
    # Filter menu data to get items from the specific restaurant
    restaurant_menu = menu_data[menu_data['restaurantName'] == restaurantName]

    # Combine user allergy and user diet
    if user_allergy or user_diet:
        # Create a combined list of allergens and diet keywords to filter
        filter_keywords = []
        if user_allergy:
            allergies = [allergy.strip().lower()
                         for allergy in user_allergy.split(',')]
            filter_keywords.extend(normalize_keywords(allergies))
        if user_diet:
            diets = [diet.strip().lower() for diet in user_diet.split(',')]
            filter_keywords.extend(normalize_keywords(diets))

        # Exclude items that contain allergens and prioritize items that match diet preferences
        restaurant_menu = restaurant_menu[
            ~restaurant_menu['itemAllergens'].str.lower().str.contains('|'.join(filter_keywords)) |
            restaurant_menu['itemDescription'].str.lower(
            ).str.contains('|'.join(filter_keywords))
        ]

    # Prioritize lower-calorie items if user prefers a low-carb or keto diet
    if user_diet and any(diet in ['low carb', 'keto'] for diet in user_diet.lower().split(',')):
        restaurant_menu = restaurant_menu.sort_values(
            by='itemNutritionalValue')

    # if user diet contains "vegetarian", recommend itemCategories or itemDescription with Vegetarian names
    # Check if the user diet includes "vegetarian"
    prioritize_vegetarian = user_diet and "vegetarian" in [
        diet.strip().lower() for diet in user_diet.split(',')]

    if prioritize_vegetarian:
        # Prioritize vegetarian items by filtering items with 'vegetarian' in category or description
        vegetarian_items = restaurant_menu[
            restaurant_menu['itemCategory'].str.lower().str.contains("vegetarian") |
            restaurant_menu['itemDescription'].str.lower(
            ).str.contains("vegetarian")
        ]

        # Prioritize bestseller vegetarian items first
        bestseller_vegetarian = vegetarian_items[vegetarian_items['itemBestSeller'] == 'Yes']
        top_items = bestseller_vegetarian.head(max_items)

        # If there are fewer than max_items, add non-bestseller vegetarian items
        if len(top_items) < max_items:
            remaining_slots = max_items - len(top_items)
            additional_vegetarian = vegetarian_items[vegetarian_items['itemBestSeller'] != 'Yes'].head(
                remaining_slots)
            top_items = pd.concat([top_items, additional_vegetarian])

        # If still fewer than max_items, add non-vegetarian items
        if len(top_items) < max_items:
            remaining_slots = max_items - len(top_items)
            non_vegetarian_items = restaurant_menu[~restaurant_menu.index.isin(
                top_items.index)]
            additional_items = non_vegetarian_items.head(remaining_slots)
            top_items = pd.concat([top_items, additional_items])
    else:
        # Prioritize bestseller items
        bestseller_items = restaurant_menu[restaurant_menu['itemBestSeller'] == 'Yes']
        # Collect top items
        top_items = bestseller_items.head(max_items)

    # If there are fewer than max_items bestsellers, fill the rest with non-bestsellers
    if len(top_items) < max_items:
        remaining_slots = max_items - len(top_items)
        non_bestsellers = restaurant_menu[restaurant_menu['itemBestSeller'] != 'Yes']
        additional_items = non_bestsellers.head(remaining_slots)
        top_items = pd.concat([top_items, additional_items])

    return top_items[['itemName', 'itemDescription', 'itemPrice', 'itemCategory', 'itemAllergens', 'itemNutritionalValue']].to_dict(orient='records')


def get_similar_restaurants(favorite, restaurant_data, tfidf_vectorizer):
    # Generate TF-IDF features for favorite restaurant
    favorite_desc = restaurant_data[restaurant_data['Restaurant']
                                    == favorite]['NameCuisine'].values[0]
    favorite_vector = tfidf_vectorizer.transform([favorite_desc]).toarray()[0]

    # Compute similarity between favorite and all other restaurants
    similarity_scores = []
    for idx, row in restaurant_data.iterrows():
        restaurant_vector = tfidf_vectorizer.transform(
            [row['NameCuisine']]).toarray()[0]
        similarity_score = cosine_similarity(favorite_vector.reshape(
            1, -1), restaurant_vector.reshape(1, -1))[0][0]
        similarity_scores.append((row['Restaurant'], similarity_score))

    # Sort by similarity score and return names of similar restaurants
    similar_restaurants = [name for name, score in sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)[:5]]
    return similar_restaurants


def user_preference_recommend(user_input, user_allergy, user_diet, n_recommendations, user_halal, user_history, user_favourites):
    restaurant_data = restDatasets
    menu_data = menuDatasets

    # Filter restaurants by halal status
    filtered_data = filter_restaurants_by_halal(restaurant_data, user_halal)

    # Build feature matrix without distance
    combined_matrix, user_cuisines, tfidf_vectorizer = build_feature_matrix(
        filtered_data, user_input)

    # Adjust the length of the user profile to match the combined matrix
    user_profile = np.zeros(combined_matrix.shape[1])

    # Add TF-IDF features to the user profile
    tfidf_user_profile = tfidf_vectorizer.transform([user_input]).toarray()[0]
    user_profile[:len(tfidf_user_profile)] = tfidf_user_profile

    # Set a single cuisine score (average of multiple preferences)
    if len(user_cuisines) > 0:
        # Set cuisine preferences (aggregated)
        user_profile[len(tfidf_user_profile)] = 1

    # Define feature weights (without distance)
    feature_weights = {
        'tfidf': 0.5,
        'cuisine': 0.3,
        'rating': 0.2
    }

    # Calculate scores for all restaurants
    scores = np.array([calculate_restaurant_score(restaurant_features, user_profile, feature_weights)
                       for restaurant_features in combined_matrix])

    # if restaurant has been recommended, then lower the score of the recommendation
    for index, restaurant in filtered_data.iterrows():
        if restaurant['Restaurant'] in user_history:
            scores[index] *= 0.8  # Apply penalty to lower the score

    # Boost scores for similar restaurants to user favorites
    for favourite in user_favourites:
        similar_restaurants = get_similar_restaurants(
            favourite, filtered_data, tfidf_vectorizer)
        for similar_restaurant in similar_restaurants:
            if similar_restaurant in filtered_data['Restaurant'].values:
                idx = filtered_data[filtered_data['Restaurant']
                                    == similar_restaurant].index[0]
                scores[idx] *= 1.2  # Boost score to prioritize

    # Get top recommendations
    top_indices = scores.argsort()[::-1][:n_recommendations]

    recommended_restaurants = []

    for index in top_indices:
        restaurant = filtered_data.iloc[index]
        score = scores[index]

        # Get top menu items
        top_menu_items = get_top_menu_items(
            restaurant['Restaurant'], menu_data, user_allergy, user_diet,)

        recommended_restaurants.append({
            'Restaurant': restaurant,
            'Score': score,
            'Top Menu Items': top_menu_items
        })
    return recommended_restaurants


def calculate_restaurant_score(restaurant_features, user_profile, feature_weights):
    # Split features into groups
    # Adjust index based on your feature order
    tfidf_features = restaurant_features[:-2]
    cuisine_features = restaurant_features[-2:-1]
    rating_feature = restaurant_features[-1]

    tfidf_score = cosine_similarity(tfidf_features.reshape(
        1, -1), user_profile[:-2].reshape(1, -1))[0][0]
    cuisine_score = np.dot(cuisine_features, user_profile[-2:-1])
    rating_score = rating_feature

    final_score = (
        feature_weights['tfidf'] * tfidf_score +
        feature_weights['cuisine'] * cuisine_score +
        feature_weights['rating'] * rating_score
    )

    return final_score
