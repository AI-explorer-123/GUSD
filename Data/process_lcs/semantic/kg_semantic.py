import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

device = 'cuda:1'


def z_score(input_tensor):
    mean = torch.mean(input_tensor, dim=0)
    std = torch.std(input_tensor, dim=0)
    normalized_tensor = (input_tensor - mean) / std
    print(normalized_tensor.shape)
    return normalized_tensor


with open('/data3/whr/zhk/Spoiler_Detection/Dataset/LCS/KG/entity2id.txt') as f:
    nodes = f.readlines()[1:]


model = AutoModel.from_pretrained(
    '/data3/whr/zhk/huggingface/bge-large-en-v1.5').to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    '/data3/whr/zhk/huggingface/bge-large-en-v1.5')

batch_size = 16
kg_semantic = torch.tensor([]).to(device)

# Process kg text
with torch.no_grad():
    for j in tqdm(range(0, len(nodes), batch_size), desc="Processing KG"):
        input = []
        for i in range(j, j+batch_size):
            if i < len(nodes):
                input.append(nodes[i].strip().split(' ')[0])

        tokens = tokenizer(input, padding=True,
                           truncation=True, return_tensors='pt').to(device)
        try:
            out = model(**tokens)[0][:, 0]
            out = torch.nn.functional.normalize(out, p=2, dim=1)
        except:
            break
        kg_semantic = torch.cat([kg_semantic, out], dim=0)

kg_semantic = z_score(kg_semantic.to(torch.float))
print(kg_semantic.shape)
torch.save(
    kg_semantic, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/final_semantic/kg_bias.pt')
