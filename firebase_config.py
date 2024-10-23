import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase app
cred = credentials.Certificate("../Desktop/mealcompass-fyp-firebase-adminsdk-y11zg-5b90887ddc.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()