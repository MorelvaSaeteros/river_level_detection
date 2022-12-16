import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
 
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import CNN
from torch.autograd import Variable



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #busca cuda, sino hay escoge CPU
print(device)


train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

print(train_data)
print(train_data.data.size())
print(train_data.targets.size())


#plt.imshow(train_data.data[0], cmap='gray')
#plt.title('%i' % train_data.targets[0])
#plt.show()

loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, #de las 60000 organiza un batch_size /en bloques de 100 para pasarle en bloques de entrenamientoq
                                          shuffle=True, 
                                          ),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          ),
}
print(loaders)

model_cnn = CNN()
print(model_cnn)

loss_func = nn.CrossEntropyLoss()   
print(loss_func)

optimizer = optim.Adam(model_cnn.parameters(), lr = 0.01)   
print(optimizer)


num_epochs = 10
def train(num_epochs, model_cnn, loaders):
    
    model_cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = model_cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
            pass
        pass
train(num_epochs, model_cnn, loaders)
torch.save(model_cnn, "model_cnn.pt")


def test():
    # Test the model
    model_cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = model_cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
        pass
test()

