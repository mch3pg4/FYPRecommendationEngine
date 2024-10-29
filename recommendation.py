import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_config import db
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Initialize Firestore DB
db = firestore.client()

# # Load datasets from Firestore
# def load_firestore_data(collection_name):
#     docs = db.collection(collection_name).stream()
#     data = []
#     for doc in docs:
#         data.append(doc.to_dict())
#     return pd.DataFrame(data)

# def load_firestore_subcollection_data(collection_name, subcollection_name):
#     docs = db.collection(collection_name).stream()
#     data = []
#     for doc in docs:
#         subcollection_docs = db.collection(collection_name).document(doc.id).collection(subcollection_name).stream()
#         for sub_doc in subcollection_docs:
#             sub_doc_data = sub_doc.to_dict()
#             sub_doc_data['parent_id'] = doc.id  # Add reference to parent document
#             data.append(sub_doc_data)
#     return pd.DataFrame(data)

# restDatasets = load_firestore_data('restaurant')
# menuDatasets = load_firestore_subcollection_data('restaurant', 'menuItems')

# Load datasets from CSV
restDatasets = pd.read_csv('GoogleReview_Penang.csv')
menuDatasets = pd.read_csv('Restaurants_Penang_Menu.csv')

# Combine restaurant name and cuisine into one column
restDatasets['NameCuisine'] = restDatasets['restaurantName'] + \
    ' ' + restDatasets['restaurantCuisine']

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
    normalized_ratings = normalize_features(
        restaurant_data['restaurantRating'].values)

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

    return top_items[['itemName']].to_dict(orient='records')


def get_similar_restaurants(favorite, restaurant_data, tfidf_vectorizer):
    # Generate TF-IDF features for favorite restaurant
    favorite_desc = restaurant_data[restaurant_data['restaurantName']
                                    == favorite]['NameCuisine'].values[0]
    favorite_vector = tfidf_vectorizer.transform([favorite_desc]).toarray()[0]

    # Compute similarity between favorite and all other restaurants
    similarity_scores = []
    for idx, row in restaurant_data.iterrows():
        restaurant_vector = tfidf_vectorizer.transform(
            [row['NameCuisine']]).toarray()[0]
        similarity_score = cosine_similarity(favorite_vector.reshape(
            1, -1), restaurant_vector.reshape(1, -1))[0][0]
        similarity_scores.append((row['restaurantName'], similarity_score))

    # Sort by similarity score and return names of similar restaurants
    similar_restaurants = [name for name, score in sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)[:5]]
    return similar_restaurants

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
        feature_weights['restaurantCuisine'] * cuisine_score +
        feature_weights['restaurantRating'] * rating_score
    )

    return final_score

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
        'restaurantCuisine': 0.3,
        'restaurantRating': 0.2
    }

    # Calculate scores for all restaurants
    scores = np.array([calculate_restaurant_score(restaurant_features, user_profile, feature_weights)
                       for restaurant_features in combined_matrix])

    # Apply penalties and boosts
    for index, restaurant in filtered_data.iterrows():
        if any(restaurant['restaurantName'] == entry['restaurantName'] for entry in user_history):
            scores[index] *= 0.8  # Apply penalty


    for favourite in user_favourites:
        similar_restaurants = get_similar_restaurants(
            favourite, filtered_data, tfidf_vectorizer)
        for similar_restaurant in similar_restaurants:
            if similar_restaurant in filtered_data['restaurantName'].values:
                idx = filtered_data[filtered_data['restaurantName']
                                    == similar_restaurant].index[0]
                scores[idx] *= 1.2  # Boost score

    # Get top recommendations
    top_indices = scores.argsort()[::-1]

    # Limit recommendations by cuisine if user has multiple preferences
    cuisine_count = {}
    recommended_restaurants = []
    total_recommendations = 0
    max_per_cuisine = n_recommendations // 2 if len(
        user_cuisines) > 1 else n_recommendations

    for index in top_indices:
        if total_recommendations >= n_recommendations:
            break

        restaurant = filtered_data.iloc[index]
        cuisine = restaurant['restaurantCuisine']

        if cuisine not in cuisine_count:
            cuisine_count[cuisine] = 0

        if cuisine_count[cuisine] < max_per_cuisine:
            score = scores[index]
            top_menu_items = get_top_menu_items(
                restaurant['restaurantName'], menu_data, user_allergy, user_diet)

            recommended_restaurants.append({
                'Restaurant': restaurant,
                'Score': score,
                'Top Menu Items': top_menu_items
            })

            cuisine_count[cuisine] += 1
            total_recommendations += 1

    return recommended_restaurants

# Function to create a user preference vector
def build_user_profile_vector(user_diet, user_cuisine, user_allergy, user_favorites):
    diet_vector = tfidf.transform([' '.join(user_diet)]).toarray()[0]
    cuisine_vector = tfidf.transform([' '.join(user_cuisine)]).toarray()[0]
    allergy_vector = tfidf.transform([' '.join(user_allergy)]).toarray()[0]
    if user_favorites:
        favorites_vector = tfidf.transform(user_favorites).toarray().sum(axis=0)
    else:
        favorites_vector = np.zeros(tfidf.transform(['']).shape[1])

    combined_vector = np.concatenate((diet_vector, cuisine_vector, allergy_vector, favorites_vector))
    return combined_vector

# Create user profiles based on preferences
def create_user_profiles(users_data):
    user_profiles = {}
    for user in users_data:
        profile_vector = build_user_profile_vector(
            user['userDiets'], user['userCuisines'], user['userAllergens'], user['favouriteRestaurants']
        )
        user_profiles[user['userId']] = profile_vector
    return user_profiles

def get_similar_users_knn(target_user_id, user_profiles, n_neighbors=5):
    # Convert user_profiles dictionary to list and keep track of user IDs
    user_ids = list(user_profiles.keys())
    profiles_matrix = np.array([user_profiles[user_id] for user_id in user_ids])
    
    # Adjust n_neighbors to the maximum possible value based on available users
    n_neighbors = min(n_neighbors, len(user_ids) - 1)
    
    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(profiles_matrix)
    
    # Get the index of the target user in user_ids
    target_index = user_ids.index(target_user_id)
    target_profile = profiles_matrix[target_index].reshape(1, -1)
    
    # Find n_neighbors most similar users (excluding the target user)
    distances, indices = knn.kneighbors(target_profile)
    
    # Retrieve user IDs of similar users
    similar_users = [user_ids[idx] for idx in indices.flatten() if user_ids[idx] != target_user_id]
    return similar_users


# hybrid recommendation function
def hybrid_recommendation(user_id, user_input, user_allergy, user_diet, n_recommendations, user_halal, user_history, user_favourites, users_data):
    # Generate collaborative recommendations
    user_profiles = create_user_profiles(users_data)
    similar_users = get_similar_users_knn(user_id, user_profiles)
    
    collaborative_scores = {}
    has_collaborative_data = False

    for similar_user in similar_users:
        recommendation_history = users_data[similar_user].get('recommendedHistory', {})
        if recommendation_history:
            has_collaborative_data = True
            for recommendation in recommendation_history:
                if recommendation not in user_history:
                    collaborative_scores[recommendation] = collaborative_scores.get(recommendation, 0) + 1

    # Adjust weight for collaborative score if data is sparse
    collaborative_weight = 0.4 if has_collaborative_data else 0
    content_weight = 1.0 if not has_collaborative_data else 0.6

    # Get top recommendations from content-based approach
    content_recommendations = user_preference_recommend(user_input, user_allergy, user_diet, n_recommendations, user_halal, user_history, user_favourites)
    
    # Combine scores using weighted average
    final_recommendations = []
    for restaurant in content_recommendations:
        score = restaurant['Score']
        collaborative_score = collaborative_scores.get(restaurant['Restaurant']['restaurantName'], 0)
        final_score = (content_weight * score) + (collaborative_weight * collaborative_score)
        
        final_recommendations.append({
            'Restaurant': restaurant['Restaurant'],
            'Score': final_score,
            'Top Menu Items': restaurant['Top Menu Items']
        })
    
    # Sort by final score
    final_recommendations.sort(key=lambda x: x['Score'], reverse=True)
    return final_recommendations[:n_recommendations]
