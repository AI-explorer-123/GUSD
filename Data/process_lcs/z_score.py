import torch

def z_score(input_tensor):
    mean = torch.mean(input_tensor, dim=0)
    std = torch.std(input_tensor, dim=0)
    normalized_tensor = (input_tensor - mean) / std
    print(normalized_tensor.shape)
    return normalized_tensor    


semantic1 = torch.load(
    '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/semantic/bge-large/all_semantic.pt', map_location='cuda:0')
semantic1 = z_score(semantic1)
torch.save(semantic1, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/normalized_semantic/bge-large/all_semantic.pt')

semantic2 = torch.load(
    '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/semantic/bge-large/part_semantic.pt', map_location='cuda:1')
semantic2 = z_score(semantic2)
torch.save(semantic2, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/normalized_semantic/bge-large/part_semantic.pt')

semantic3 = torch.load('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/semantic/e5-large/all_semantic.pt', map_location='cuda:2')
semantic3 = z_score(semantic3)
torch.save(semantic3, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/normalized_semantic/e5-large/all_semantic.pt')

semantic4 = torch.load('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/semantic/e5-large/part_semantic.pt', map_location='cuda:3')
semantic4 = z_score(semantic4)
torch.save(semantic4, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/normalized_semantic/e5-large/part_semantic.pt')
