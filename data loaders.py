# %%

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Step 2: Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, label = self.data[index], self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# Step 4: Create an instance of your custom dataset
# Assuming you have 'data' and 'targets' as your custom data
custom_data = CustomDataset(data, targets, transform=transforms.ToTensor())

# Step 5: Create a DataLoader
batch_size = 32
shuffle = True

custom_dataloader = DataLoader(custom_data, batch_size=batch_size, shuffle=shuffle)

# Now you can iterate over custom_dataloader to get batches of data during training or evaluation


# %%
import torch
from torch.utils.data import Dataset, DataLoader


# Step 1: Define a custom dataset class
class CustomTensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


# Step 2: Create an instance of your custom dataset
# Assuming you have 5 tensors named tensor1, tensor2, tensor3, tensor4, tensor5
custom_dataset = CustomTensorDataset(tensor1, tensor2, tensor3, tensor4, tensor5)

# Step 3: Create a DataLoader
batch_size = 16
shuffle = True

custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)

# Now you can iterate over custom_dataloader to get batches of data during training or evaluation
