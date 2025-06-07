import torch
import json

with open('/data3/whr/zhk/Spoiler_Detection/Dataset/LCS/KG/train2id.txt') as f:
    lines = f.readlines()[1:]

genre = [[] for _ in range(147191)]

for line in lines:
    start, end, rel = line.strip().split(' ')
    if rel == '14':
        genre[int(start)-494394].append(int(end)-494365)

print(genre[:10])
with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/map/movie_genre.json', 'w') as f:
    json.dump(genre, f, indent=4)
