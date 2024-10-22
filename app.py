# use flask to create web api for recommendation system to be called to android
from flask import Flask, request, jsonify
from recommendation import user_preference_recommend

# create app
app = Flask(__name__)


@app.route("/", methods=['GET'])
def recommend():
    try:
        # data = request.json
        # user_input = data.get('user_input')
        # user_diet = data.get('user_diet')
        # user_allergy = data.get('user_allergy')
        # n_recommendations = data.get('n_recommendations', 5)

        user_input = "Vegetarian"
        user_halal = "No"
        user_diet = "Keto"
        user_allergy = "Dairy"
        user_favourites = [
            'Rempah Ratus - Banana Leaf Cafe with South Indian Cuisine (Indian Restaurant, Banana Leaf Rice, Briyani, Claypot and Curry)']
        user_history = [
            'Rempah Ratus - Banana Leaf Cafe with South Indian Cuisine (Indian Restaurant, Banana Leaf Rice, Briyani, Claypot and Curry)']
        n_recommendations = 9

        # Use your existing recommendation function
        recommended_restaurants = user_preference_recommend(
            user_input, user_allergy, user_diet, n_recommendations, user_halal, user_history, user_favourites)

        recommendations_list = []
        for recommendations in recommended_restaurants:
            restaurant = recommendations['Restaurant']
            score = recommendations['Score']
            menu_items = recommendations['Top Menu Items']

            restaurant_info = {
                "Restaurant": restaurant['Restaurant'],
                "Cuisine": restaurant['Cuisine'],
                "Rating": restaurant['Rating'],
                "Halal": restaurant['isHalal'],
                "Score": score,
                "Top Menu Items": [
                    {
                        "itemName": item['itemName'],
                        "itemDescription": item['itemDescription'],
                        "itemPrice": item['itemPrice'],
                        "itemNutritionalValue": item['itemNutritionalValue']
                    } for item in menu_items
                ]
            }
            recommendations_list.append(restaurant_info)

        return jsonify(recommendations_list), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


app.add_url_rule('/recommend', 'recommend', recommend, methods=['GET'])


if __name__ == '__main__':
    app.run(debug=True)
