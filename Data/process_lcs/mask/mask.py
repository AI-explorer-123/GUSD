import torch

train_mask = [i+406896 for i in range(0, 1302500)]
valid_mask = [i+406896 for i in range(1302500, 1674643)]
test_mask = [i+406896 for i in range(1674643, 1860715)]

train_mask = torch.Tensor(train_mask)
valid_mask = torch.Tensor(valid_mask)
test_mask = torch.Tensor(test_mask)

torch.save(train_mask, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/mask/train_mask.pt')
torch.save(valid_mask, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/mask/valid_mask.pt')
torch.save(test_mask, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/mask/test_mask.pt')
