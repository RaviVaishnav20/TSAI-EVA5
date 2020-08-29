from utils import *
dropout_value = 0.01     #.01


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            # output_image = 26, RF=3
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(dropout_value)
        )

        # Transition BLOCK 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            # output_image = 26, RF=3
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(dropout_value)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            # output_image = 24, RF=5
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),
            nn.Dropout2d(dropout_value)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # output_image = 12, RF=6

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            # output_image = 10, RF=10
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(dropout_value)
        )

        # TRANSITION BLOCK 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # output_image = 10, RF=10
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),
            nn.Dropout2d(dropout_value)
        )

        # CONVOLUTION BLOCK 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            # output_image = 8, RF=14
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),
            nn.Dropout2d(dropout_value)
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            # output_image = 8, RF=20
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(dropout_value)
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            # output_image = 6, RF=24
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(dropout_value)
        )

        # GAP Layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6),

        )

        # FC layer
        self.trans3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # output_image = 6, RF=28
            # nn.ReLU()  NEVER!!!!
            # nn.BatchNorm2d(num_features=10)     NEVER!!!!
            # nn.Dropout2d(dropout_value)    NEVER!!!!
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.trans1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.trans2(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.trans3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
