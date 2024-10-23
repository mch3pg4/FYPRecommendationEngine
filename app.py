# use flask to create web api for recommendation system to be called to android
from flask import Flask, request, jsonify
from recommendation import user_preference_recommend
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_config import db

# create app
app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return "Welcome to MealCompass Recommendation!"


@app.route('/user', methods=['GET'])
# get firebase data
def get_firebase_data():
    db = firestore.client()
    users_ref = db.collection(u'users')
    docs = users_ref.stream()

    user_list = []
    for doc in docs:
        # if userType is user, add only user preferences stuff to user_list
        if doc.to_dict()['userType'] == 'user':
            doc_data = doc.to_dict()
            user = {
                # get doc id
                "userId": doc.id,
                "userDiets": doc_data['userDiets'],
                "userCuisines": doc_data['userCuisines'],
                "userAllergens": doc_data['userAllergens'],
                "favouriteRestaurants": doc_data['favouriteRestaurants'],
                "recommendedHistory": doc_data['recommendedHistory']
            }
            user_list.append(user)

    return jsonify(user_list), 200


@app.route('/restaurant', methods=['GET'])
def get_restaurant_and_menu_data():
    db = firestore.client()
    restaurant_ref = db.collection(u'restaurant')
    docs = restaurant_ref.stream()

    restaurant_list = []
    for doc in docs:
        doc_data = doc.to_dict()
        restaurant_id = doc.id
        restaurant = {
            "restaurantId": restaurant_id,
            "restaurantName": doc_data['restaurantName'],
            "menuItems": []
        }

        # Get menu items for each restaurant
        subcollection = restaurant_ref.document(
            restaurant_id).collection(u'menuItems')
        menu_docs = subcollection.stream()
        for menu_item in menu_docs:
            menu_item_data = menu_item.to_dict()
            menu = {
                "itemId": menu_item.id,
                "itemName": menu_item_data['itemName'],
                "itemPrice": menu_item_data['itemPrice'],
            }
            restaurant["menuItems"].append(menu)

        restaurant_list.append(restaurant)

    return jsonify(restaurant_list), 200


@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        # get current user id
        user_id = "DRQlrqYsR2TEPbFsMixvyULNxZs2"

        # get user data from firebase
        db = firestore.client()
        users_ref = db.collection(u'users')
        docs = users_ref.stream()

        restaurant_ref = db.collection(u'restaurant')

        # retrieve user data from user_id
        user_data = {}
        for doc in docs:
            if doc.id == user_id:
                doc_data = doc.to_dict()
                user_data = {
                    "userDiets": doc_data['userDiets'],
                    "userCuisines": doc_data['userCuisines'],
                    "userAllergens": doc_data['userAllergens'],
                    "favouriteRestaurants": doc_data['favouriteRestaurants'],
                    "recommendedHistory": doc_data['recommendedHistory']
                }

        # add vegetarian in also if userdiets contain vegetarian
        if "Vegetarian" in user_data['userDiets']:
            user_data['userCuisines'].append("Vegetarian")
        user_input = user_data['userCuisines']
        user_halal = "Yes" if "Halal" in user_data['userDiets'] else "No"
        user_diet = user_data['userDiets']
        user_allergy = user_data['userAllergens']

        # get restaurant name from id
        user_favourites = []
        if user_data['favouriteRestaurants'] is not None:
            for restaurant_id in user_data['favouriteRestaurants']:
                restaurant_doc = restaurant_ref.document(restaurant_id).get()
                if restaurant_doc.exists:
                    user_favourites.append(
                        restaurant_doc.to_dict()['restaurantName'])

        user_history = []
        if user_data['recommendedHistory'] is not None:
            for restaurant_id in user_data['recommendedHistory']:
                restaurant_doc = restaurant_ref.document(restaurant_id).get()
                if restaurant_doc.exists:
                    user_history.append(
                        restaurant_doc.to_dict()['restaurantName'])

        n_recommendations = 5

        # convert user input from list to string
        user_input = ', '.join(user_input)
        user_halal = ', '.join(user_halal)
        user_diet = ', '.join(user_diet)
        user_allergy = ', '.join(user_allergy)

        # recommendation function
        recommended_restaurants = user_preference_recommend(
            user_input, user_allergy, user_diet, n_recommendations, user_halal, user_history, user_favourites)

        recommendations_list = []
        for recommendations in recommended_restaurants:
            restaurant = recommendations['Restaurant']
            score = recommendations['Score']
            menu_items = recommendations['Top Menu Items']

            restaurant_info = {
                "Restaurant": restaurant['restaurantName'],
                "Score": score,
                "Top Menu Items": [
                    {
                        "itemName": item['itemName'],
                    } for item in menu_items
                ]
            }
            recommendations_list.append(restaurant_info)

        return jsonify(recommendations_list), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
