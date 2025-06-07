import torch
import json

print("Loading data...")
users = json.load(
    open('/data4/zhk/Spoiler_Detection/Dataset/LCS/user.json'))
movie_id = json.load(
    open('/data4/zhk/Spoiler_Detection/Data/processed_lcs_data/movie_id.json'))
print("Loading finished.")

user_idx = 147191
review_movie_map = [0] * 406896
review_user_map = [0] * 406896

for user in users.values():
    for review in user['reviews']:
        review_user_map.append(user_idx)
        review_movie_map.append(movie_id[review['movie_id']])
    user_idx += 1

review_user_map = torch.Tensor(review_user_map)
print(review_user_map.shape, review_user_map[:10])
torch.save(review_user_map,
           '/data4/zhk/Spoiler_Detection/Data/processed_lcs_data/map/review_user_map.pt')

review_movie_map = torch.Tensor(review_movie_map)
print(review_movie_map.shape, review_movie_map[:10])
torch.save(review_movie_map,
           '/data4/zhk/Spoiler_Detection/Data/processed_lcs_data/map/review_movie_map.pt')
