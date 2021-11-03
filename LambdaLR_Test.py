import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np

NUM_EPOCHS = 100

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
scheduler = LambdaLR(optimizer=optimizer,
                     lr_lambda=lambda epoch: 1.0 if epoch < 10 else np.math.exp(0.1 * (10 - epoch)))

lr_history = []
for epoch in range(NUM_EPOCHS):
    optimizer.step()
    lr_history.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(range(len(lr_history)), lr_history)
plt.title('LambdaLR')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('learning rate')
plt.show()
