import pandas as pd
import datetime
import json
import random
from tqdm import tqdm
import numpy as np

reviews = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/Kaggle/fixed_review.json'))
movies = json.load(open(
    '/data3/whr/zhk/Spoiler_Detection/Dataset/Kaggle/IMDB_movie_details_new.json'))
movie_id = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/movie_id.json'))
users = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/user_review_new.json'))


# def get_year(movie):
#     try:
#         year = int(movie['startYear'])
#     except:
#         try:
#             year = int(movie['endYear'])
#         except:
#             year = random.randint(1874, 2022)
#     return year


n1 = []
n2 = []
timestamp = []

user_idx = 1572

for user, reviews_idx in tqdm(users.items(), desc='Processing users and reviews...'):
    for review_idx in reviews_idx:
        review = reviews[review_idx]
        ts = ((pd.to_datetime(
            [review['review_date']]) - pd.Timestamp("1874-01-01")) // pd.Timedelta("1D"))[0]
        n1.append(user_idx)
        n2.append(review_idx)
        timestamp.append(ts)
        
        n1.append(review_idx)
        n2.append(user_idx)
        timestamp.append(ts)

        n1.append(movie_id[review['movie_id']])
        n2.append(review_idx)
        timestamp.append(ts)
    user_idx += 1

timestamp = np.array(timestamp)
origin = np.min(timestamp)
timestamp = (timestamp-origin).tolist()

label = [0] * len(n1)

data = {'u': n1,
        'i': n2,
        'ts': timestamp,
        'label': label}
data = pd.DataFrame(data=data)

data.sort_values(by='ts', ascending=True, inplace=True)
data.reset_index(drop=True, inplace=True)

data['idx'] = data.index+1
data.to_csv(
    '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/time.csv')
