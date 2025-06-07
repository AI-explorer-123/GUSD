import torch
import json
from tqdm import tqdm

print('Loading data...')
nodes = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/node.json'))
print('Loading finished.')

all_meta = []
review_idx = 259705 + 147191

for node in tqdm(nodes):
    if node['type'] == 'movie':
        try:
            meta = torch.tensor([int(node['feature']['meta']['year']),
                                int(node['feature']['meta']['isAdult']),
                                int(node['feature']['meta']['runtime']),
                                int(node['feature']['meta']['rating']),
                                int(node['feature']['meta']['numVotes']),
                                0])
        except:
            meta = torch.tensor([int(node['feature']['meta']['year']),
                                 int(node['feature']['meta']['isAdult']),
                                 0,
                                 int(node['feature']['meta']['rating']),
                                 int(node['feature']['meta']['numVotes']),
                                 0])
    elif node['type'] == 'user':
        review_avg_length = 0
        for _ in range(node['feature']['meta']['review count']):
            review_avg_length += len(nodes[review_idx]['feature']['semantic'])
            review_idx += 1

        review_avg_length /= node['feature']['meta']['review count']
        meta = torch.tensor([int(node['feature']['meta']['badge count']),
                             int(node['feature']['meta']['review count']),
                             int(node['feature']['meta']['avg_helpful']),
                             int(node['feature']['meta']['avg_total']),
                             int(node['feature']['meta']['avg_score']),
                             review_avg_length])
    elif node['type'] == 'review':
        meta = torch.tensor([int(node['feature']['meta']['helpful vote count']),
                             int(node['feature']['meta']['total vote count']),
                             int(node['feature']['meta']['score']),
                             0, 0, 0])
    else:
        raise NameError(f"There is no {node['type']}-type node in the data!")

    all_meta.append(meta)

all_meta = torch.stack(all_meta, dim=0).to(torch.float)


def z_score(input_tensor):
    mean = torch.mean(input_tensor, dim=0)
    std = torch.std(input_tensor, dim=0)
    normalized_tensor = (input_tensor - mean) / std
    print(normalized_tensor.shape)
    return normalized_tensor


all_meta = z_score(all_meta)
print(all_meta.shape)
torch.save(all_meta, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/meta.pt')
