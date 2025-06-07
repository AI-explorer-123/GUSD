import json
reviews = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/Kaggle/IMDB_reviews.json'))
users = {}
for idx, review in enumerate(reviews):
    if users.get(review['user_id']) == None:
        users[review['user_id']] = [idx]
    else:
        users[review['user_id']].append(idx)
with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/user_review.json', 'w') as f:
    json.dump(users, f, indent=4)
