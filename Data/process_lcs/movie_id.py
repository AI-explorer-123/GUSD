import torch
import json

movies = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/LCS/movie.json'))
dic = {}
for id, movie in enumerate(movies.keys()):
    dic[movie] = id

with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/movie_id.json', 'w') as f:    
    json.dump(dic, f, indent=4)
