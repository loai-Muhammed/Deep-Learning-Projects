import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), #224x224
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),nn.ReLU(),   
            nn.Conv2d(16, 32, 3,padding=1), #112x112
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),#112x112
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),#112x112
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),#112x112
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),#56x56
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),#28x28
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),#14x14
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            #nn.Conv2d(384, 256, 3, padding=1),#7x7
            #nn.MaxPool2d(2, 2),
            #nn.ReLU(),
        )
        self.flat = nn.Flatten()
        
        self.head = nn.Sequential(
            nn.Linear(512*7*7, 512),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, 84),
            nn.Dropout(p=dropout),
            nn.Linear(84, num_classes)
        )
            
            
                                      
                                      
                                      
                                      
        '''
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) #224x224
        self.relu1 = nn.ReLU()
        self.batnor1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) #112x112
        #self.drop2 = nn.Dropout2d(p=dropout)
        self.batnor2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  #56x56
        #self.drop3 = nn.Dropout2d(p=dropout)
        self.batnor3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  #28x28
        #self.drop4 = nn.Dropout2d(p=dropout)
        self.batnor4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)   
        
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)  #14x14
        #self.drop5 = nn.Dropout2d(p=dropout)
        self.batnor5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2) 
        
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)  #7x7
        #self.drop6 = nn.Dropout2d(p=dropout)
        self.batnor6 = nn.BatchNorm2d(1024)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(2, 2) #3x3
        
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024*3*3, 512)  
        self.dp1 = nn.Dropout(p=dropout)
        self.batn1 = nn.BatchNorm1d(512)
        self.rl1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 256)  
        #self.dp2 = nn.Dropout(p=dropout)
        self.batn2 = nn.BatchNorm1d(256)
        #self.rl2 = nn.ReLU()
        
        self.fc3 = nn.Linear(256, 150)
        self.dp3 = nn.Dropout(p=dropout)
        self.rl3 = nn.ReLU()
        
        self.fc4 = nn.Linear(150, 84)
        #self.dp4 = nn.Dropout(p=dropout)
        self.rl4 = nn.ReLU()
        
        self.fc5 = nn.Linear(84, num_classes)
        self.fc6 = nn.LogSoftmax(dim=1)
        '''

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        x = self.backbone(x)
        x = self.flat(x)
        x = self.head(x)
        '''
        x = self.relu1(self.pool1(self.batnor1(self.conv1(x))))
        x = self.relu2(self.pool2(self.batnor2(self.conv2(x))))
        x = self.relu3(self.pool3(self.batnor3(self.conv3(x))))
        x = self.relu4(self.pool4(self.batnor4(self.conv4(x))))
        x = self.relu5(self.pool5(self.batnor5(self.conv5(x))))
        x = self.relu6(self.pool6(self.batnor6(self.conv6(x))))

        x = self.flatten(x)

        x = self.rl1(self.batn1(self.dp1(self.fc1(x))))
        x = self.batn2(self.fc2(x))
        #x = self.rl2(self.batn2(self.dp2(self.fc2(x))))
        x = self.rl3(self.dp3(self.fc3(x)))
        x = self.rl4(self.fc4(x))
        
        x = self.fc5(x)
        x = self.fc6(x)
        '''

        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
