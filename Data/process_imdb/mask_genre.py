import torch
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

all_genre = json.load(open(
    '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/review_genre.json'))
reviews_genre = []
review_genre_num = []
for review_genre in tqdm(all_genre):
    review_genre_num.append(len(review_genre))
    review_genre = torch.Tensor(review_genre)+1
    review_genre = F.pad(review_genre, pad=(0, 3-review_genre.size(0)))
    reviews_genre.append(review_genre)
review_genre_num = torch.Tensor(review_genre_num)
reviews_genre = torch.stack(reviews_genre, dim=0)
review_genre = review_genre.to(torch.int64)
print(review_genre_num.dtype)
# torch.save(review_genre_num,
#            '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/review_genre_num.pt')
torch.save(reviews_genre,
           '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/review_genre.pt')


all_genre = json.load(open(
    '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/movie_genre.json'))
movies_genre = []
movie_genre_num = []
for movie_genre in tqdm(all_genre):
    movie_genre_num.append(len(movie_genre))
    movie_genre = torch.Tensor(movie_genre)+1
    movie_genre = F.pad(movie_genre, pad=(0, 3-movie_genre.size(0)))
    movies_genre.append(movie_genre)
movies_genre = torch.stack(movies_genre, dim=0)
movie_genre_num = torch.Tensor(movie_genre_num)
movie_genre = movie_genre.to(torch.int64)
# print(movie_genre_num.shape)
print(movie_genre_num.dtype)
# torch.save(movie_genre_num,
#            '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/movie_genre_num.pt')
torch.save(movies_genre,
           '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/movie_genre.pt')
