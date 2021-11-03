import torch
from torch import nn
from torch.optim.lr_scheduler import CyclicLR
import matplotlib.pyplot as plt

NUM_EPOCHS = 100

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0)
scheduler = CyclicLR(optimizer=optimizer,
                     base_lr=1e-5, max_lr=1e-1, step_size_up=10, mode='exp_range', gamma=0.9)

lr_history = []
for epoch in range(NUM_EPOCHS):
    optimizer.step()
    lr_history.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(range(len(lr_history)), lr_history)
plt.title('CyclicLR-exp_range')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('learning rate')
plt.show()
