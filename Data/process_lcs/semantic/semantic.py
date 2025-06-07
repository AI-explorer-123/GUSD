import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

device = "cuda:0"


def z_score(input_tensor):
    mean = torch.mean(input_tensor, dim=0)
    std = torch.std(input_tensor, dim=0)
    normalized_tensor = (input_tensor - mean) / std
    print(normalized_tensor.shape)
    return normalized_tensor


print('Loading data...')
nodes = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/node.json'))
print('Loading finished.')

model = AutoModel.from_pretrained(
    '/data3/whr/zhk/huggingface/roberta-large').to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    '/data3/whr/zhk/huggingface/roberta-large')

batch_size = 16
semantic = torch.tensor([]).to(device)
# part_semantic = torch.tensor([]).to(device)
# node_type = []

# Process all text
with torch.no_grad():
    for j in tqdm(range(0, len(nodes), batch_size), desc="Processing all"):
        input = []
        for i in range(j, j+batch_size):
            if i < len(nodes):
                input.append(nodes[i]['feature']['all'])
            #     if nodes[i]['type'] == 'movie':
            #         node_type.append(0)
            #     elif nodes[i]['type'] =='user':
            #         node_type.append(1)
            #     else:
            #         node_type.append(2)

        tokens = tokenizer(input, max_length=512, padding='max_length',
                           truncation=True, return_tensors='pt').to(device)
        out = model(**tokens)['last_hidden_state'].mean(1)
        # out = accelerator.gather(out)
        semantic = torch.cat([semantic, out], dim=0)

torch.save(
    semantic, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/final_semantic/roberta/raw_all_semantic.pt')
semantic = z_score(semantic.to(torch.float))
print(semantic.shape)
torch.save(semantic, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/final_semantic/roberta/all_semantic.pt')

# node_type = torch.Tensor(node_type)
# torch.save(
#     node_type, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/node_type.pt')
# print(node_type.shape)

# Process part text
# with torch.no_grad():
#     for j in tqdm(range(0, len(nodes), batch_size), desc="Processing part"):
#         input = []
#         for i in range(j, j+batch_size):
#             if i < len(nodes) :
#                 if nodes[i]['type'] != 'user':
#                     input.append(nodes[i]['feature']['semantic_meta'])
#                 else :
#                     input.append(nodes[i]['feature']['all'])
#         try:
#             tokens = tokenizer(input, padding=True,
#                             truncation=True, return_tensors='pt').to('cuda:0')
#             out = model(**tokens)[0][:, 0]
#             out = torch.nn.functional.normalize(out, p=2, dim=1)
#             part_semantic = torch.cat([part_semantic, out], dim=0)
#         except:
#             pass


# part_semantic = z_score(part_semantic.to(torch.float))
# print(part_semantic.shape)
# torch.save(
#     part_semantic, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/final_semantic/bge-large/meta_semantic.pt')
