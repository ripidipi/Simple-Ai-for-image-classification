import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
import pytorch_save_and_load_test
from pathlib import Path
from torch import nn
start_time = time.time()


"""00 Some base operation"""
"""#1 Tensor""" 
"""1 Way how to create tensor""" 
### scalar 
#scalar = torch.tensor(7)
#print(scalar.ndim)  #Get a quantity of axes
#print(scalar)       
#print(scalar.item)  #Get tensor back as PY int 
#print(scalar.size)  #Array size       


### vector 
#vector = torch.tensor([7, 7])
#print(vector)
#print(vector.shape) #Return quantity of element in all axis  


### create a random tensor of size 
#random_tensor = torch.rand(3, 4)
#print(random_tensor)


### create a random tensor with similar shape to an image tensor
#random_image_size_tensor = torch.rand(size=(3, 224, 224))    # colour channels (R, G, B), height, width, 
#print(random_image_size_tensor)
#print(random_image_size_tensor.shape, random_image_size_tensor.ndim)


### zeros and ones tensor 
#zeros = torch.zeros(size=(3, 4))
#ones = torch.ones(size=(3, 4))
#print(zeros.dtype, ones.dtype)

 

"""2creating a range of tensors and tensors-like """
### use torch.range()
#one_to_ten = torch.range(start=1, end=100, step=8)
#one_to_nine = torch.arange(1, 10, 2)
#print(one_to_nine)
#print(one_to_ten)


###creating tensor like
#ten_zeros = torch.zeros_like(input=one_to_ten)
#print(ten_zeros)



"""3 tensor datatypes"""
### float 32 tensor 
"""
float_32_tensor = torch.tensor([3.0, 6., 9.],
							  dtype=None, #choose datatype 
							  device=None, #device that using for calculations it could be "cuda", "cpu"
							  requires_grad=False)  #whether or not to track gradients with this tensors operations
float_16_tensor = float_32_tensor.type(torch.float16)
int_32_tensor = torch.tensor([3, 6, 9], 
							 dtype=torch.int32)
print(float_16_tensor * int_32_tensor)
"""

###find out details about some tensor 
"""
some_tensor = torch.rand(3, 4, device="cuda")
print(some_tensor)
print(f"Datetype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device tensor is on: {some_tensor.device}")
"""



"""4 manipulating tensors"""
##tensor operation include
# * addition 
# * subtraction 
# * multiplication 
# * devision 
# * matrix multiplication 

#tensor = torch.tensor([[1, 2, 3],
#                      [3, 4, 5]])
#print(tensor * 11)
#print(torch.add(tensor, 17))  #it's better 



"""5 matrix multiplication """
###element wish multiplication
#print(tensor)
#print(f"Equals: {tensor * tensor}")   #i know that it work like element multipling
#print(torch.matmul(tensor.T, tensor))     #it work good
#print(torch.rand(2, 5) @ torch.rand(5, 67))
# TORCH.MM  same with torch.matmul


##finding the min, max, mean.sum, etc
#x = torch.arange(0, 100, 11)
#rand_x = torch.rand(2, 3, 4)
#print(rand_x)
#print(torch.min(rand_x))
#print(rand_x[0])
#print(torch.mean(x.type(torch.float16)))
#print(x.type(torch.float16).mean()) #it work too
#print(x.sum(), torch.sum(x))


##find the positional min and max 
#print(rand_x[0].argmax()) #get index of corect argument (argmax/min)


##reshaping, stacking, squeezing and unsqueezing tensor 
#x = torch.arange(1., 10.)
#print(x, x.shape) 

##add an extra dimansion 
#x_reshaped = x.reshape(3, 3, 1)
#print(x_reshaped, x_reshaped.shape)

##change the view
#z = x.view(3, 3) 
#print(z, z.shape)
#z[0][0] = 5
#x_reshaped[2][2] = 5
#print(z.ndim, x)

##stack tensors on top of each other 
#x_stacked = torch.stack([x_reshaped, x_reshaped], dim=1)
#print(x_stacked)


##squeeze - removes all single dimensions 

#x_sqeeze = x_reshaped.squeeze()

#print(x_sqeeze, x_sqeeze.shape)


##unsqueeze - adds a single dimensions at specific dim
#x_unsqueezed = x_sqeeze.unsqueeze(dim=1)
#print(x_unsqueezed, x_unsqueezed.shape, x_sqeeze, x_sqeeze.shape)


#x_original = torch.rand(3, 224, 224)
#x_permuted = x_original.permute(2, 0, 1)
#print(x_original.shape , x_permuted.shape) 



"""6 Indexing (selecting data from tensor) """
#x = torch.arange(1, 10).reshape(1, 3, 3)
#print(x, x.shape)
#print(x[:, 1, 1])

#array = np.arange(1.0, 8.0)
#tensor = torch.from_numpy(array).type(torch.float16)
#print(array, tensor)

#tensor = torch.ones(7)
#numpy_tensor = tensor.numpy()
#print(tensor, numpy_tensor.dtype)


"""7 Reproducbility (trying to take random out of random)"""
#let made a seed like in minecraft but for tensor
#you mast use manual_seed after all tensor to get them it
random_seed = 42
#torch.manual_seed(random_seed)
#random_tensor_a = torch.rand(3,4)
#torch.manual_seed(random_seed)
#random_tensor_b = torch.rand(3,4)
#print(random_tensor_a, random_tensor_b)
#print(random_tensor_a == random_tensor_b)

#running tensors and pythorch objact on the GPUs 
#GPUs =  faster computation on numbers, thanks to CUDA 


"""#2 Work with GPU"""
#print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using : {device}")
#print(torch.cuda )
#tensor = torch.tensor([1, 2, 3])
#tensor = torch.tensor([1, 2, 3], device=device_cuda)
#tensor_gpu = tensor.to(device_cuda)
#tensor_back_to_cpu = tensor_gpu.cpu().numpy()
#print(tensor_back_to_cpu)
#print(tensor_gpu) 
#print(tensor_gpu.numpy()) we couldn't as np haven't suported np
#seed = 234188
#torch.manual_seed(seed)
#ex = torch.matmul(torch.rand(7,7), torch.arange(21).type(torch.float32).reshape(7, 3))
#print(ex)







"""01 Pytorch workflow
what_were_covering = 1. data(prepare and load)
					 2. build model
					 3. fitting the model to data(training)
					 4. making predictions and evaluting a model (inference) 
					 5. saving and loading model
					 6. putting it all together
"""
#TORCH.NN is the basic building blocks for graphs
#print(torch.__version__)




"""1. Data """
# Machine learning is a game of two parts:
# 1 get data into a numberical representation 
# 2 build a model to learn patterns in that numrtical representation 
#create *Known* for parameters 
weight = 0.7
bias = 0.3

# create 
start = 0 
end = 1
step = 0.02 
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
#print(X[:10], y[:10], len(X), len(y))


"""
Splitting data into training and tests sets 
(one of the most important concepts in machine learning in general)
"""

#Create a train/test split 
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
#print(len(X_train), len(y_train), len(X_test), len(y_test))

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

#plot_predictions()




"""2. Build model """
#create linear regration model class 
class LinearRegressionModel(nn.Module): #<- everything in pytorch 
	
	def __init__(self):
		super().__init__()
		self.weights = nn.Parameter(torch.randn(1,
												requires_grad=True,
												dtype=torch.float))
		self.bias = nn.Parameter(torch.randn(1,
												requires_grad=True,
												dtype=torch.float))
		self.path = 'models/01_pytorch_model_0.pth' 

	#forward method to define the computation in the model
	def forward(self, x: torch.Tensor) -> torch.Tensor: 
		#linear regration operation 
		return self.weights * x + self.bias 

# create a rand seed
torch.manual_seed(42)
# create instance of the model (this is a subclass of nn.Modele)
model_0 = LinearRegressionModel()
#model_0.to(device)
#print(list(model_0.parameters()))
#print(next(model_0.parameters()).device)
with torch.inference_mode():
	y_preds = model_0(X_test)

#print(y_preds)
#plot_predictions(predictions=y_preds)

"""
What our model does 
 # 1. start with random value 
 # 2. look at training data and adjust the random values to better represent 
 # (or get closer to) the ideal values (the weiht & bias value we used 
 # to create the data)
# how does it do so? 
 #through two main algorithms:
 # 1. Gradient descent
 # 2. Backpropagation

*torch.nn - contains all of the byildings for computational graphs
 ( nn can be consident a computational graphs)
*torch.nn.Parameter - what parameters our model should  try and learn,
 pytorch layer from torch.nn will set these for us 
*torch.nn.Module - the base class for all neural network modules, 
 if you sunbclass it, you should overwrite forward
*torch.optim - this where the optimizers in PyTorch live, they will help with gradient 
*def forward() - all nn.Model require you to overwrite forward(), 
 this what will be forward in computation

*the whole idea of training is for a model to move from some "unknown" parameters
 (it could be random)to some "known" parameters.
*or in other words from a poor representation of the data to better 
 representation of the data.
*one way to measure how poor or how wrong your models predictions are is 
 to use a loss function.
*Note: loss function may also be called cost function or criterion in different 
	areas. for our case, we're going to refer to it as a loss function.

Things that we need to train:
*LOSS FUNCTION - a function to measure how wrong your model's predictions are 
 to the ideal outputs 
*OPTIMIZER - takes into account the loss of a model and adjust the model's 
 parameters (e.g weight & biase) to improve the loss function.
* Only for pytorch we need
 a train loop 
 a test loop
"""
#print(list(model_0.parameters()))
#print(model_0.state_dict())


#Setup a loss function
loss_fn = nn.L1Loss() 


#Serup an optimizer 
optimizer = torch.optim.SGD(params=model_0.parameters(),
							lr=0.01) #lr - learning  rate = posible the most important hyperparamrter that you can set 


#Go building a training and testing loop
 # 0. loop throught the 
 # 1. forward pass(this involves data moving through our model's forward() fn) to make predictions on data 
 # 2. calculate the loss(compare forward pass predictions to ground truth labels)
 # 3. optimizer zero grade
 # 4. loss backward - move backwards through the network to calculate the gradients of each of 
 # the parameters of our model with respect to the loss **backpropagation**
 # 5. optimazer step - use the optimizer to adjust our model's parameters to try and improve the loss **gradient desent**

# An epoch is one loop through the data...
epochs = 200
epoch_count = []
loss_value = []
test_loss_values = []

# 0. loop through the data
for epoch in range(epochs):
	
	model_0.train() # train mode in pytorch sets all parameters that required gradients to required gradients 
	#set the model to training mode
	
	#1. forward pass
	y_preds = model_0(X_train)

	# 2. calculate the loss
	loss = loss_fn(y_preds, y_train)

	# 3. optimizer zero grad 
	optimizer.zero_grad()

	# 4. Perform backpropagation on the loss with respect to the parameters of the model 
	loss.backward()

	# 5. Step the optimizer (perform gradient descent)
	optimizer.step() #by deault how the optimizer changes will acculumate through the loop so... 
	 #we have to zero them above in step 3 for the next iteration of the loop 

	# Settings in the model not needed for evalution/testing (dropout/batch norm layers)
	model_0.eval() #turn off gradient tracking

	with torch.inference_mode():
		# do the forward pass
		test_preds = model_0(X_test)
		# calculate the loss
		test_loss = loss_fn(test_preds, y_test)
	# Print out What's Happenin'
	if epoch % 20 == 0:
		epoch_count.append(epoch)
		loss_value.append(loss)
		test_loss_values.append(test_loss)
		#print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
#print(loss)
#rint(model_0.state_dict())
with torch.inference_mode():
	y_preds_new = model_0(X_test)

#plot_predictions(predictions=y_preds_new)
def making_plot_after_lossfn(epochs, loss_val, test_loss_val):
	plt.plot(epochs, np.array(torch.tensor(loss_val).cpu().numpy()), label="Train loss")
	plt.plot(epochs, test_loss_val, label="Test loss")
	plt.title("Training and test loss cuves")
	plt.ylabel("LOSS")
	plt.xlabel("EPOCHS")
	plt.legend()
	plt.show()


#making_plot_after_lossfn(epoch_count, loss_value, test_loss_values)

"""
Saving and loading model in PyTorch
#three main method you should for saveing and loading models in PyTorch
1. "torch.save" - allows you save a Pytorch object in Python's pikle format.  
2. "torch.load" - allows you load a saved pytorch object
3. "torch.nn.Module.load_state_dict()" - this allows to load a model's saved state dictionary
"""

### Saving a model from pytorch 

pytorch_save_and_load_test.save(model_0)

### Load a model to pytorch
# since we saved our model's "state_dict()" rether the entire model, we'll create a new 
# instance of our model class and load the the save "state_dict" into that

# to load in a saved data we have to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=model_0.path))
#loaded_model_0.to(device)
#print("loaded", loaded_model_0.state_dict())

loaded_model_0.eval()
with torch.inference_mode():
	loaded_model_preds = loaded_model_0(X_test)

# print(loaded_model_preds == y_preds_new)

# creatre some data using the linear regration formula of y = weight * X + bias

class LinearRegrassionModelV2(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		# Use nn.Leanear() for creating the model parameters
		self.linear_layer = nn.Linear(in_features=1,
										out_features=1)
		self.path = 'models/01_pytorch_model_1.pth'

	def forward(self, x: torch.Tensor):
		return self.linear_layer(x)
	
torch.manual_seed(42)

model_1 = LinearRegrassionModelV2()
#loaded_model_0.to(device)
#print(model_1, model_1.state_dict())
# new instance of lineral regrasion

pytorch_save_and_load_test.save(model_1)

loaded_model_1 = LinearRegrassionModelV2()
loaded_model_1.load_state_dict(torch.load(f=model_1.path))
#loaded_model_1.to(device)
print(next(loaded_model_1.parameters()).device)
# print(loaded_model_1.state_dict())
# print(next(loaded_model_0.parameters()).device)
# print(loaded_model_0.state_dict())
# print(y_preds)
loaded_model_1.eval()
with torch.inference_mode():
	loaded_model_1_preds = loaded_model_1(X_test)

print(loaded_model_1_preds)
print(y_preds)
print(loaded_model_preds == loaded_model_1_preds)



print(time.time() - start_time)