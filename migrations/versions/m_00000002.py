from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['tabular-wizard']

# Creating or getting the collection
users_collection = db['aiModels']

# Creating indexes
# Ensure an index on 'email' field, similar to the unique constraint in SQL
users_collection.create_index([('name', 1)], unique=True)

print("MongoDB setup script completed.")