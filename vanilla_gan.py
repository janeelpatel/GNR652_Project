import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from matplotlib import pyplot as plt

def mnist_data():
    compose = transforms.Compose([transforms.ToTensor()])
    return datasets.MNIST(root='/data/', train=True, transform=compose, download=True)

# Load data
data = mnist_data()
# Batch Size
batchsize = 100
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=True)
# Num batches
num_batches = len(data_loader)

# three hidden-layer discriminator neural network
class DiscriminatorNet(torch.nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
	
# three hidden-layer generator neural network
class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
	
# function converts images to vectors
def images_to_vectors(images):
    return images.view(images.size(0), 784)

# function converts vectors to images
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

# function generates random noise matrix of Size : size x 100
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda() 
    return n

# function generates tensor containing ones, with shape = size
def real_data_target(size):
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

# function generates tensor containing zeroes, with shape = size
def fake_data_target(size):
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

# function for training discriminator
def train_discriminator(discriminator, generator, optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

# function for training generator
def train_generator(discriminator, generator, optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

# initialize discriminator and generator net 
discriminator = DiscriminatorNet()
generator = GeneratorNet()
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()
    
# define optimizers to be used for respective nets
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Binary Cross-Entropy loss function
loss = nn.BCELoss()

# Number of epochs for training
num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):

        # 1. for training discriminator
        real_data = Variable(images_to_vectors(real_batch))
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train Discriminator
        d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator, generator, d_optimizer,
                                                                real_data, fake_data)

        # 2. for training generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train Generator
        g_error = train_generator(discriminator, generator, g_optimizer, fake_data)

    # Display Progress
    test_images = vectors_to_images(generator(noise(1))).data.cpu()
    plt.imshow(test_images.view(28,28)) # display one generated image per epoch
    plt.show()
    print("Epoch = " + "{}/{}".format(epoch+1, num_epochs))
    print("D_Err = ", round(d_error.item(),3)) # Discriminator Error for current epoch
    print("G_Err = ", round(g_error.item(),3)) # Generator Error for current epoch
