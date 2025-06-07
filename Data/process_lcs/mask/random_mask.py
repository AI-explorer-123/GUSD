import torch
import random

numbers = list(range(406896, 406896+215779))

set1 = random.sample(numbers, 151045)

remaining_numbers = list(set(numbers) - set(set1))
set2 = random.sample(remaining_numbers, 172623-151045)

remaining_numbers = list(set(remaining_numbers) - set(set2))
set3 = random.sample(remaining_numbers, 215779-172623)

train_mask = torch.Tensor(list(set1))
print(train_mask.shape)
valid_mask = torch.Tensor(list(set2))
print(valid_mask.shape)
test_mask = torch.Tensor(list(set3))
print(test_mask.shape)

torch.save(train_mask, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/random_mask/train_mask.pt')
torch.save(valid_mask, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/random_mask/valid_mask.pt')
torch.save(test_mask, '/data3/whr/zhk/Spoiler_Detection/Data/processed_data/random_mask/test_mask.pt')