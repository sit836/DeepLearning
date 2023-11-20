import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


lr = 1e-4
batch_size = 256
epochs = 2

# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()
dataset = datasets.MNIST(root="./data",
                         train=True,
                         download=True,
                         transform=tensor_transform)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=True)

model = AE()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

outputs = []
losses = []
for epoch in tqdm(range(epochs)):
    for (image, _) in loader:
        # Reshaping the image to (-1, 784)
        image = image.reshape(-1, 28 * 28)

        # Output of Autoencoder
        reconstructed = model(image)

        # Calculating the loss function
        loss = loss_function(reconstructed, image)

        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().numpy())
        outputs.append((epochs, image, reconstructed))

# plt.plot(losses)
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.show()

for image_i, recon_i in zip(image, reconstructed):
    image_i = image_i.reshape(-1, 28, 28)
    recon_i = recon_i.reshape(-1, 28, 28)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(image_i[0].detach().numpy())
    ax2.imshow(recon_i[0].detach().numpy())
    plt.show()
