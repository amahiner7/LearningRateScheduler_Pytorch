import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

NUM_EPOCHS = 100

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)

lr_history = []
for epoch in range(NUM_EPOCHS):
    optimizer.step()
    lr_history.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(range(len(lr_history)), lr_history)
plt.title('ExponentialLR')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('learning rate')
plt.show()


