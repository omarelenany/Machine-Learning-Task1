# Install surprise library if you haven't already
# pip install scikit-surprise

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Load the dataset
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1, 5))
data = Dataset.load_from_file('your_dataset.csv', reader=reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use user-based collaborative filtering
sim_options = {
    'name': 'cosine',  # Compute similarities between users
    'user_based': True  # Use user-based collaborative filtering
}

# Initialize the KNNBasic algorithm
algo = KNNBasic(sim_options=sim_options)

# Train the algorithm on the training set
algo.fit(trainset)

# Get top recommendations for a user
user_id = 'user_id_to_recommend_for'
# Get a list of items the user has not rated
items_to_predict = [item for item in data.df['item'].unique() if item not in data.df[data.df['user'] == user_id]['item'].unique()]
# Predict ratings for these items
predictions = [algo.predict(user_id, item) for item in items_to_predict]
# Sort predictions by estimated rating
top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]

# Print top recommendations
for prediction in top_predictions:
    print('Song:', prediction.iid, 'Rating:', prediction.est)