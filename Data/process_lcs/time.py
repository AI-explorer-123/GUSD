import pandas as pd
import datetime
import json
import random
from tqdm import tqdm
import numpy as np

movies = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/LCS/movie.json'))
movie_id = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/movie_id.json'))
users = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/LCS/user.json'))


def get_year(movie):
    try:
        year = int(movie['startYear'])
    except:
        try:
            year = int(movie['endYear'])
        except:
            year = random.randint(1874, 2022)
    return year


n1 = []
n2 = []
timestamp = []

for id, movie in tqdm(movies.items()):
    year1 = get_year(movie)
    for similar_movie_id in movie['similar_movie']:
        if movie_id.get(similar_movie_id) != None:
            year2 = get_year(movies[similar_movie_id])
            n1.append(movie_id[id])
            n2.append(movie_id[similar_movie_id])
            timestamp.append(
                ((pd.to_datetime([f"1 January {max(year1, year2)}"]) - pd.Timestamp("1874-01-01")) // pd.Timedelta("1D"))[0])

user_idx = 147191
review_idx = 406896

for user in tqdm(users.values()):
    for review in user['reviews']:
        ts=((pd.to_datetime(
            [review['time']]) - pd.Timestamp("1874-01-01")) // pd.Timedelta("1D"))[0]
        n1.append(user_idx)
        n2.append(review_idx)
        timestamp.append(ts)
        
        n1.append(movie_id[review['movie_id']])
        n2.append(review_idx)
        timestamp.append(ts)

        # n1.append(user_idx)
        # n2.append(movie_id[review['movie_id']])

        # timestamp.append(ts)        

        review_idx += 1
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
data.to_csv('/data3/whr/zst/DyGLib/processed_data/zhk2/ml_zhk.csv')
