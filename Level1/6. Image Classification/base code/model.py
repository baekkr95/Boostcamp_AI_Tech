import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)



# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 18, bias=True)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)

        return x

### Multi Label Model
class MultiBranchModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.base_model = torchvision.models.resnet18(pretrained=True)
        # classifier 전까지 사용
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        self.mask = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512, 3, bias=True)
        )
        self.gender = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512, 2, bias=True)
        )
        self.age = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512, 3, bias=True)
        )
        

    def forward(self, x):
        x = self.base_model(x)
        # print(x.shape)

        x = torch.flatten(x, start_dim=1)
        # print('22: ', x.shape)

        return {
            'mask': self.mask(x),
            'gender': self.gender(x),
            'age': self.age(x)
        }

    # Loss 함수 구현 부분
    def get_loss(self, net_output, ground_truth):
        mask_loss = F.cross_entropy(net_output['mask'], ground_truth['mask'].to(device))
        gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender'].to(device))
        age_loss = F.cross_entropy(net_output['age'], ground_truth['age'].to(device))

        loss = mask_loss + gender_loss + age_loss

        return loss #, {'mask' : mask_loss, 'gender' : gender_loss, 'age' : age_loss}