import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

NUM_EPOCHS = 100

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer=optimizer,
                              mode='min', factor=0.5, min_lr=-1e-7, patience=10)

lr_history = []
for epoch in range(NUM_EPOCHS):
    optimizer.step()
    lr_history.append(optimizer.param_groups[0]['lr'])

    val_loss = 0.1
    scheduler.step(val_loss)

plt.plot(range(len(lr_history)), lr_history)
plt.title('ReduceLROnPlateau')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('learning rate')
plt.show()
