import torch
import json
from tqdm import tqdm

print("Loading data...")
nodes = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/node.json'))

users = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Dataset/LCS/user.json'))
print('Loading finished.')

for i, user in tqdm(enumerate(users.values())):
    all_helpful = 0
    all_votes = 0
    all_score = 0
    
    for review in user['reviews']:
        all_helpful += review['helpful']
        all_votes += review['helpful_total']
        all_score += review['point']
        
    try:
        avg_helpful = all_helpful/len(user['reviews'])
        avg_votes = all_votes/len(user['reviews'])
        avg_score = all_score/len(user['reviews'])
    except:
        avg_helpful = 0
        avg_votes = 0
        avg_score = 0

    nodes[i+147191]['feature']['meta']['avg_helpful'] = avg_helpful
    nodes[i+147191]['feature']['meta']['avg_total'] = avg_votes
    nodes[i+147191]['feature']['meta']['avg_score'] = avg_score
    nodes[i+147191]['feature']['all'] = f"The badge count of the user is {len(user['badges'].split('|'))-1}, which are {user['badges']}.The review count of the user is {len(user['reviews'])}. An average of {avg_votes} people voted for each of the review, and {avg_votes} users think the review is helpful on average. The average point of the reviews is {avg_score}."

with open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/changed_nodes.json', 'w') as f:
    json.dump(nodes, f, indent=4)
