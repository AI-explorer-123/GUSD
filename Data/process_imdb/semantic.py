import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def z_score(input_tensor):
    mean = torch.mean(input_tensor, dim=0)
    std = torch.std(input_tensor, dim=0)
    normalized_tensor = (input_tensor - mean) / std
    print(normalized_tensor.shape)
    return normalized_tensor


device = 'cuda:3'

print('Loading data...')
nodes = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/nodes.json'))
print('Loading finished.')

model = AutoModel.from_pretrained(
    '/data3/whr/zhk/huggingface/bge-large-en-v1.5').to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    '/data3/whr/zhk/huggingface/bge-large-en-v1.5')

batch_size = 32
all_semantic = torch.tensor([]).to(device)

# Process all text
with torch.no_grad():
    for j in tqdm(range(0, len(nodes), batch_size), desc="Processing all"):
        input = []
        for i in range(j, j+batch_size):
            if i < len(nodes):
                input.append(nodes[i]['feature']['all'])

        tokens = tokenizer(input, padding=True,
                           truncation=True, return_tensors='pt').to(device)
        try:
            out = model(**tokens)[0][:, 0]
            out = torch.nn.functional.normalize(out, p=2, dim=1)
        except:
            break
        all_semantic = torch.cat([all_semantic, out], dim=0)

all_semantic = z_score(all_semantic.to(torch.float))
print(all_semantic.shape)
torch.save(
    all_semantic, '/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/final_semantic/bge-large/all_semantic.pt')
