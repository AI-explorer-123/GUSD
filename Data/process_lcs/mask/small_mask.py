import torch
import json

train_mask = [i+406896 for i in range(0, 7000)]
valid_mask = [i+406896 for i in range(7000, 9000)]
test_mask = [i+406896 for i in range(9000, 10000)]

train_mask = torch.Tensor(train_mask)
valid_mask = torch.Tensor(valid_mask)
test_mask = torch.Tensor(test_mask)

torch.save(
    train_mask, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/small_mask/train_mask.pt')
torch.save(
    valid_mask, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/small_mask/valid_mask.pt')
torch.save(
    test_mask, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/small_mask/test_mask.pt')
