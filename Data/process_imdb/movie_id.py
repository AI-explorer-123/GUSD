import torch
import json

movies = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/Kaggle/IMDB_movie_details_new.json'))
dic = {}
for id, movie in enumerate(movies):
    dic[movie['movie_id']] = id

with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/movie_id.json', 'w') as f:
    json.dump(dic, f, indent=4)
