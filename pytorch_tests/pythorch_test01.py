import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
import pytorch_save_and_load_test
from pathlib import Path
from torch import nn
start_time = time.time()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using : {device}")

weight = 0.967 
bias = 0.3

start = 0
end = 1
step = 1e-2


X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

#print(X[:10], y[:10])

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train), len(y_train), len(X_test), len(y_test))

def plot_predictions(train_data=X_train,
					train_lables=y_train,
					test_data=X_test,
					test_lables=y_test,
					predictions=None):

	
	#Plot training data, test data and compares prediction
	
	plt.figure(figsize=(10, 7))
	#scatter - разброс
	#plot training data in blue
	plt.scatter(train_data, train_lables, c="b", s=4, label='training data')
	#plot testing data in red
	plt.scatter(test_data, test_lables, c="r", s=4, label='tasting data')
	
	#are there predictions?
	if predictions is not None:
		#if predictions is exist plot it 
		plt.scatter(test_data, predictions, c="g", s=5, label='Predictions')

	plt.legend(prop={'size': 10});
	plt.show()

# plot_predictions(X_train, y_train, X_test, y_test)

class LinearRegrationModelV2(nn.Module):
	
    def __init__(self) -> None:
        super().__init__()
        self.lineral_layer = nn.Linear(in_features=1, 
                                       out_features=1)
        self.path = 'models/01_pythorch_model_1.pth'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
          return self.lineral_layer(x)


model_1 = LinearRegrationModelV2()
# model_1.to(device)

# print(next(model_1.parameters()).device)
# print(model_1, model_1.state_dict())

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params = model_1.parameters(), 
                            lr=1e-2 - 1e-3 * 5)

epochs = int(1e3 * 5)


for epoch in range(epochs):
      model_1.train()

      # Forward pass
      y_pred = model_1(X_train)

      # Calculate the loss 
      loss = loss_fn(y_pred, y_train)

      # Optimazer zero grad 
      optimizer.zero_grad()

      # Perform backpropogation
      loss.backward()

      # otimaze the step
      optimizer.step()

      model_1.eval()
      with torch.inference_mode():
            test_pred = model_1(X_test)
            test_loss = loss_fn(test_pred, y_test)
      #if epoch % 100 == 0:
            #print(f'Loss: {loss} Test loss {test_loss} \n') 

model_1.eval()        
with torch.inference_mode():
      y_preds = model_1(X_test)


# print(model_1.state_dict())

# plot_predictions(predictions=y_preds.cpu())

pytorch_save_and_load_test.save(model_1)
loaded_model_1 = LinearRegrationModelV2()
loaded_model_1.load_state_dict(torch.load(f=model_1.path))
loaded_model_1.eval()
with torch.inference_mode():
	loaded_model_1_preds = loaded_model_1(X_test)

# plot_predictions(predictions=loaded_model_1_preds.cpu())







print(time.time() - start_time)