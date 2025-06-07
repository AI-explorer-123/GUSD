import torch
import json
import numpy as np
import pandas as pd
users = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/user_review_new.json'))
movies = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/Kaggle/IMDB_movie_details_new.json'))
movie_id = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/movie_id.json'))
reviews = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/Kaggle/fixed_review.json'))

genre = []
num = []
for user_idx, reviews_idx in users.items():
    for review_id in reviews_idx:
        review = reviews[review_id]
        try:
            movie = movies[movie_id[review['movie_id']]]
            genre.append(movie['genre'])
            num.append(len(movie['genre']))
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
# with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/review_genre.json', 'w') as f:
#     json.dump(reviews_genre, f, indent=4)
