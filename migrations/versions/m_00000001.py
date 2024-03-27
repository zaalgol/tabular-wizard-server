from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['tabular-wizard']

# Creating or getting the collection
users_collection = db['users']

# Creating indexes
# Ensure an index on 'email' field, similar to the unique constraint in SQL
users_collection.create_index([('email', 1)], unique=True)

# There's no direct equivalent to creating columns as in SQL since MongoDB is schema-less
# But we can ensure that the 'email' field is indexed and unique
# Any other schema enforcement or default values would need to be handled at the application level or through MongoDB validation rules

print("MongoDB setup script completed.")