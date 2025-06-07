import torch
import json
import numpy as np
import pandas as pd
users = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/LCS/user.json'))
movies = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/LCS/movie.json'))

genre = []
num = []
for id, user in users.items():
    for review in user['reviews']:
        try:
            genre.append(movies[review['movie_id']]
                         ['genres'].strip().split(','))
            num.append(len(movies[review['movie_id']]
                           ['genres'].strip().split(',')))
        except:
            genre.append(['None'])
            num.append(1)
genre_list = [st for sublists in genre for st in sublists]
all_genre = np.array(genre_list)
genre_uni, genre_mapping = np.unique(all_genre, return_inverse=True)
print(genre_uni.shape, genre_uni)
# genre_mapping = genre_mapping.tolist()
# reviews_genre = []
# idx = 0
# for num_i in num:
#     reviews_genre.append(genre_mapping[idx:idx+num_i])
#     idx += num_i
# with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/review_genre.json', 'w') as f:
#     json.dump(reviews_genre, f, indent=4)