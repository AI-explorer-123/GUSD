import torch
import json
from tqdm import tqdm
import random
import pandas as pd

print('Loading raw data...')
movies = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/Kaggle/IMDB_movie_details_new.json'))
all_reviews = json.load(open('/data3/whr/zhk/Spoiler_Detection/Dataset/Kaggle/IMDB_reviews.json'))
users = json.load(open(
    '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/user_review.json'))
movie_id = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/movie_id.json'))

print('Loading finished.')

movie_idx = 0
user_idx = 1572
review_start = 264979

nodes = [None]*838892
base_edge_index = []
base_edge_type = []

review_edge_index = []
review_edge_type = []

label = [0]*573913

for movie in tqdm(movies, desc='Processing movies...'):
    semantic_meta = ''

    try:
        time_delta = pd.to_timedelta(time_str)
        minutes = time_delta.total_seconds() / 60
        runtime = movie['duration']
        semantic_meta += f'The movie duration is {runtime} minutes.'
    except:
        runtime = random.randint(1, 180)

    semantic_meta += f"The rating of this movie is {movie['rating']}."

    try:
        semantic_meta = semantic_meta + \
            f"The genres of the movie are: {','.join(movie['genre'])}"
    except:
        ...

    if movie['plot_summary'] != None:
        all = f"The summary of the movie is:'{movie['plot_summary']}'" + \
            semantic_meta
    elif movie['plot_synopsis'] != None:
        all = f"The synopsis of the movie is:'{movie['plot_synopsis']}'" + \
            semantic_meta
    else:
        all = semantic_meta

    movie_node = {
        'id': movie_idx,
        'type': 'movie',
        'feature':
            {
                'semantic': movie['plot_summary'],
                'meta':
                {
                    'runtime': runtime,
                    'rating': movie['rating'],
                    'genres': movie['genre'],
                },
                "semantic_meta": semantic_meta,
                "all": all
            }
    }
    nodes[movie_idx] = movie_node
    movie_idx += 1

for user, reviews in tqdm(users.items(), desc='Processing users and reviews...'):
    all_rating = 0
    all_length = 0

    for review_idx in reviews:
        review = all_reviews[review_idx]
        all_rating += int(review['rating'])
        all_length += len(review['review_text'])
        review_node = {
            "id": review_idx,
            "type": "review",
            "movie_id": review['movie_id'],
            "feature": {
                "semantic": review['review_text'],
                "meta": {
                    "rating": review['rating'],
                    "length": len(review['review_text'])
                },
                "semantic_meta": f"The review was published on {review['review_date']}.The rating of the review is {review['rating']}.",
                "all": f"The review is:'{review['review_text']}'The review was published on {review['review_date']}. The rating of the review is {review['rating']}."
            }
        }
        nodes[review_start+review_idx] = review_node

        base_edge_index.append(torch.tensor(
            [movie_id[review['movie_id']], review_idx]))
        base_edge_type.append(0)

        base_edge_index.append(torch.tensor(
            [user_idx, review_idx]))
        base_edge_type.append(3)

        base_edge_index.append(torch.tensor(
            [review_idx, user_idx]))
        base_edge_type.append(1)

        review_edge_index.append(torch.tensor(
            [review_idx, movie_id[review['movie_id']]]))
        review_edge_type.append(2)

        if review['is_spoiler'] == True:
            label[review_idx] = 1

    avg_rating = all_rating/len(reviews)
    avg_length = all_length/len(reviews)
    user_node = {
        "id": user_idx,
        "type": "user",
        "feature": {
            "all": f"The review count of the user is {len(reviews)}. The average rating of the reviews is {avg_rating}.And the average length of the reviews of the users is {avg_length}.",
            "meta": {
                "review count": len(reviews),
                'avg_rating': avg_rating,
                'avg_length': avg_length,
            }
        }
    }
    nodes[user_idx] = user_node
    user_idx += 1

print('Saving data...')
with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/nodes.json', 'w') as f:
    json.dump(nodes, f, indent=4)
print(len(nodes))
base_edge_index = torch.stack(base_edge_index, dim=1)
torch.save(base_edge_index,
           '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/edge/base_edge_index.pt')
print(base_edge_index.shape)

base_edge_type = torch.Tensor(base_edge_type)
torch.save(base_edge_type,
           '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/edge/base_edge_type.pt')
print(base_edge_type.shape)

review_edge_index = torch.stack(review_edge_index, dim=1)
torch.save(review_edge_index,
           '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/edge/review_edge_index.pt')
print(review_edge_index.shape)

review_edge_type = torch.Tensor(review_edge_type)
torch.save(review_edge_type,
           '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/edge/review_edge_type.pt')
print(review_edge_type.shape)

label = torch.Tensor(label)
torch.save(
    label, '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/label.pt')
print(label.shape)

print('Saving finished.')
