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

print("Loading...")
model = AutoModel.from_pretrained(
    '/data3/whr/zhk/huggingface/bge-large-en-v1.5').to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    '/data3/whr/zhk/huggingface/bge-large-en-v1.5')

expl = json.load(
    open('/data3/whr/zhk/Spoiler_Detection/Data/processed_data/llm/expl.json'))
print("Loading finished.")

batch_size = 16
user_bias_semantic = torch.tensor([]).to(device)

# Process expl
with torch.no_grad():
    for j in tqdm(range(0, len(expl), batch_size), desc="Processing expl"):
        input = []
        for i in range(j, j+batch_size):
            if i < len(expl):
                input.append(expl[i])

        tokens = tokenizer(input, padding=True,
                           truncation=True, return_tensors='pt').to(device)
        try:
            out = model(**tokens)[0][:, 0]
            out = torch.nn.functional.normalize(out, p=2, dim=1)
        except:
            break
        user_bias_semantic = torch.cat([user_bias_semantic, out], dim=0)

user_bias_semantic = z_score(user_bias_semantic.to(torch.float))
print(user_bias_semantic.shape)
torch.save(
    user_bias_semantic, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/llm/user_bias.pt')
