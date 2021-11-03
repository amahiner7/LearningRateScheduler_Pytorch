import torch
from torch import nn
from custom_scheduler.CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
import matplotlib.pyplot as plt

NUM_EPOCHS = 100

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0)
scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer,
                                          T_0=20, T_mult=1, eta_max=1e-1, T_up=10, gamma=0.5)

lr_history = []
for epoch in range(NUM_EPOCHS):
    optimizer.step()
    lr_history.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(range(len(lr_history)), lr_history)
plt.title('CosineAnnealingWarmUpRestarts')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('learning rate')
plt.show()
