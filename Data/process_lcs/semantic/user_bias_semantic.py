import torch
import json
from tqdm import tqdm

print('Loading data.')
all_semantic = torch.load(
    '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/semantic/bge-large/part_semantic.pt').to('cuda:3')
nodes = json.load(open(
    '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/node.json'))
print("Loading finished.")

user_start = 147191
review_start = 259705 + 147191
review_idx = 259705 + 147191

users_semantic = []

for user_idx in tqdm(range(user_start, review_start)):
    user = nodes[user_idx]
    user_semantic = []
    review_avg_length = 0
    for i in range(user['feature']['meta']['review count']):
        review_semantic = all_semantic[review_idx].squeeze().unsqueeze(0)
        user_semantic.append(review_semantic)
        review_avg_length += len(nodes[review_idx]['feature']['semantic'])
        review_idx += 1
    user_semantic = torch.stack(user_semantic, dim=0).mean(dim=0)
    users_semantic.append(user_semantic)

users_semantic = torch.cat(users_semantic, dim=0)
print(users_semantic.shape)
torch.save(users_semantic,
           '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/final_semantic/user_bias_semantic.pt')
