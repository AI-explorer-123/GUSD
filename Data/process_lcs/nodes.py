import torch
import json
from tqdm import tqdm
import random

print('Loading raw data...')
movies = json.load(
    open('/data4/zhk/Spoiler_Detection/Dataset/LCS/movie.json'))
users = json.load(
    open('/data4/zhk/Spoiler_Detection/Dataset/LCS/user.json'))
movie_id = json.load(
    open('/data4/zhk/Spoiler_Detection/Data/processed_lcs_data/movie_id.json'))

print('Loading finished.')

movie_idx = 0
user_idx = 147191
review_idx = 406896

nodes = [None]*2267611
similar_movie_edge_index = []
similar_movie_edge_type = []

base_edge_index = []
base_edge_type = []

review_edge_index = []
review_edge_type = []

label = [0] * 406896

for movie in tqdm(movies.values(), desc='Processing movies...'):
    semantic_meta = ''
    try:
        year = int(movie['startYear'])
        semantic_meta += f'The movie was first released in {year}.'
    except:
        try:
            year = int(movie['endYear'])
        except:
            year = random.randint(1874, 2022)
    try:
        isAdult = movie['isAdult']
        judge = "is" if isAdult == True else "is not"
        semantic_meta += f'The movie {judge} an adult movie.'
    except:
        isAdult = random.randint(0, 1)
    try:
        runtime = movie['runtimeMinutes']
        semantic_meta += f'The movie duration is {runtime} minutes.'
    except:
        runtime = random.randint(1, 180)

    semantic_meta += f"A total of {movie['numVotes']} people voted for this movie, with an average score of {movie['rating']}."
    try:
        semantic_meta = semantic_meta + f"Keywords: {movie['genres']}"
    except:
        ...

    if movie['summary'] != None:
        all = f"The summary of the movie is:'{movie['summary']}'" + \
            semantic_meta
    elif movie['synopsis'] != None:
        all = f"The synopsis of the movie is:'{movie['synopsis']}'" + \
            semantic_meta
    else:
        all = semantic_meta
    movie_node = {
        'id': movie_idx,
        'type': 'movie',
        'feature':
            {
                'semantic': f"{movie['summary']}",
                'meta':
                {
                    'year': year,
                    'isAdult': isAdult,
                    'runtime': runtime,
                    'rating': movie['rating'],
                    'numVotes': movie['numVotes']
                },
                "semantic_meta": semantic_meta,
                "all": all
            }
    }
    nodes[movie_idx] = movie_node

    for similar_movie in movie['similar_movie']:
        if movie_id.get(similar_movie) != None:
            similar_movie_edge_index.append(torch.tensor(
                [movie_idx, movie_id[similar_movie]]))
            similar_movie_edge_type.append(4)

    movie_idx += 1

for user in tqdm(users.values(), desc='Processing users...'):
    all_helpful = 0
    all_votes = 0
    all_score = 0

    for review in user['reviews']:
        all_helpful += review['helpful']
        all_votes += review['helpful_total']
        all_score += review['point']

    try:
        avg_helpful = all_helpful/len(user['reviews'])
        avg_votes = all_votes/len(user['reviews'])
        avg_score = all_score/len(user['reviews'])
    except:
        avg_helpful = 0
        avg_votes = 0
        avg_score = 0
        
    user_node = {
        "id": user_idx,
        "type": "user",
        "feature": {
            "all": f"The badge count of the user is {len(user['badges'].split('|'))-1}, which are {user['badges']}.The review count of the user is {len(user['reviews'])}. An average of {avg_votes} people voted for each of the review, and {avg_votes} users think the review is helpful on average. The average point of the reviews is {avg_score}.",
            "meta": {
                "badge count": len(user['badges'].split('|'))-1,
                "review count": len(user['reviews']),
                'avg_helpful': avg_helpful,
                'avg_total': avg_votes,
                'avg_score':avg_score
            }
        }
    }
    nodes[user_idx] = user_node

    for review in user['reviews']:
        review_node = {
            "id": review_idx,
            "type": "review",
            "movie_id": review['movie_id'],
            "feature": {
                "semantic": review['content'],
                "meta": {
                    "helpful vote count": review['helpful'],
                    "total vote count": review['helpful_total'],
                    "score": review['point']
                },
                "semantic_meta": f"The review was published on {review['time']}. A total of {review['helpful_total']} people voted for this review, and {review['helpful']} users think it is helpful. The point of the review is {review['point']}.",
                "all": f"The review is:'{review['content']}'The review was published on {review['time']}. A total of {review['helpful_total']} people voted for this review, and {review['helpful']} users think it is helpful. The point of the review is {review['point']}."
            }
        }
        nodes[review_idx] = review_node

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

        if review['spoiler'] == True:
            label.append(1)
        else:
            label.append(0)

        review_idx += 1

    user_idx += 1

# print('Saving data...')
# with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/node.json', 'w') as f:
#     json.dump(nodes, f, indent=4)
# print(len(nodes))

# similar_movie_edge_index = torch.stack(similar_movie_edge_index, dim=1)
# torch.save(similar_movie_edge_index,
#            '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/edge/similar_movie_edge_index.pt')
# print(similar_movie_edge_index.shape)

# similar_movie_edge_type = torch.Tensor(similar_movie_edge_type)
# torch.save(similar_movie_edge_type,
#            '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/edge/similar_movie_edge_type.pt')
# print(similar_movie_edge_type.shape)

# base_edge_index = torch.stack(base_edge_index, dim=1)
# torch.save(base_edge_index,
#            '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/edge/base_edge_index.pt')
# print(base_edge_index.shape)

# base_edge_type = torch.Tensor(base_edge_type)
# torch.save(base_edge_type,
#            '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/edge/base_edge_type.pt')
# print(base_edge_type.shape)

# review_edge_index = torch.stack(review_edge_index, dim=1)
# torch.save(review_edge_index,
#            '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/edge/review_edge_index.pt')
# print(review_edge_index.shape)

# review_edge_type = torch.Tensor(review_edge_type)
# torch.save(review_edge_type,
#            '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/edge/review_edge_type.pt')
# print(review_edge_type.shape)

label = torch.Tensor(label)
torch.save(label, '/data4/zhk/Spoiler_Detection/Data/processed_lcs_data/label.pt')
print(label.shape)

print('Saving finished.')
