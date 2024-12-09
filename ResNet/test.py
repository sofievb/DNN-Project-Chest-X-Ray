
import torch
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=None)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-2)

for i, param_group in enumerate(optimizer.param_groups):
    print(f"Parameter Group {i + 1}: {param_group}")
