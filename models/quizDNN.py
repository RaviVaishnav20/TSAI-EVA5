from utils import *
K_S = 128


class QuizDNN(nn.Module):
    def __init__(self, dropout=0.05):
        super(QuizDNN, self).__init__()
        self.dropout_val = dropout

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=K_S, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(K_S)
        self.conv2 = nn.Conv2d(in_channels=K_S, out_channels=K_S, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(K_S)
        self.conv3 = nn.Conv2d(in_channels=K_S, out_channels=K_S, kernel_size=(3, 3), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(K_S)
        self.pool = nn.MaxPool2d(2, 2)

        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=K_S, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2 + x1)))
        x3 = self.pool(x3)

        x5 = F.relu(self.bn3(self.conv3(x3)))
        x6 = F.relu(self.bn3(self.conv3(x5 + x3)))
        x7 = F.relu(self.bn3(self.conv3(x6 + x5 + x3)))
        x7 = self.pool(x7)

        x9 = F.relu(self.bn3(self.conv3(x7)))
        x10 = F.relu(self.bn3(self.conv3(x7 + x9)))
        x11 = F.relu(self.bn3(self.conv3(x7 + x9 + x10)))

        x12 = self.gap_linear(x11)
        x12 = x12.view(-1, 10)

        return x12
