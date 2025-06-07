import torch
import json
import numpy as np
import pandas as pd
movies = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/Kaggle/IMDB_movie_details_new.json'))

genre = []
num = []
for movie in movies:
    try:
        genre.append(movie['genre'])
        num.append(len(movie['genre']))
    except:
        genre.append(['None'])
        num.append(1)
genre_list = [st for sublists in genre for st in sublists]
all_genre = np.array(genre_list)
genre_uni, genre_mapping = np.unique(all_genre, return_inverse=True)
print(genre_uni.shape, genre_uni)
genre_mapping = genre_mapping.tolist()
movie_genre = []
idx = 0
for num_i in num:
    movie_genre.append(genre_mapping[idx:idx+num_i])
    idx += num_i
# with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/movie_genre.json', 'w') as f:
#     json.dump(movie_genre, f, indent=4)
