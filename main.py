

#loss를 그래프로 표시하는 방법과 다양한 opt 함수를 테스트 해본다.
#가장 그래프가 잘 나오는 opt 함수를 선택한다.
#pytorch loss 설명
#https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7

import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np

#cuda로 보낸다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('========VGG 테스트 =========')
print("========입력데이터 생성 [batch, color, image x, image y]=========")
#이미지 사이즈를 어떻게 잡아도 vgg는 다 소화한다.


#그래프 저장용
# loss_graph = []
# iter_graph = []


#==========================================
#1) 모델 생성
model = models.vgg16(pretrained=True).to(device)


print(model)




print("========맨 뒤에 모듈 추가=========")
my_layer2 = nn.Sigmoid()
model.classifier.add_module("7", my_layer2)


print(model)

print('========= Summary로 보기 =========')
#Summary 때문에 cuda, cpu 맞추어야 함
#뒤에 값이 들어갔을 때 내부 변환 상황을 보여줌
#adaptive average pool이 중간에서 최종값을 바꿔주고 있음
summary(model, (3, 100, 100))


#2) loss function
#꼭 아래와 같이 2단계, 클래스 선언 후, 사용
# criterion = nn.MSELoss()
# criterion = nn.SmoothL1Loss()
# criterion = nn.L1Loss()
# criterion = nn.Tanh()
# criterion = nn.KLDivLoss()

# criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.MarginRankingLoss()
# criterion = nn.CosineEmbeddingLoss()


#3) activation function
learning_rate = 1e-4
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.002)

lossList = []
optimizerList = []

lossList.append(nn.MSELoss())
lossList.append(nn.L1Loss())
lossList.append(nn.SmoothL1Loss())

optimizerList.append(torch.optim.Adamax(model.parameters(), lr=0.002))
# optimizerList.append(torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9))
optimizerList.append(torch.optim.Adagrad(model.parameters(), lr =0.001))
optimizerList.append(torch.optim.Adam(model.parameters(), lr=1e-4))

# loss_graph = [[] for _ in range(len(lossList) * len(optimizerList))]  # 각 조합에 대한 loss 그래프 초기화
loss_graph = [[] for _ in range(len(lossList) * len(optimizerList))]  # 각 조합에 대한 loss 그래프 초기화
# iter_graph = [[] for _ in range(len(lossList) * len(optimizerList))]  # 각 조합에 대한 iteration 그래프 초기화
iter_graph = []  # 각 조합에 대한 iteration 그래프 초기화

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


#모델이 학습 모드라고 알려줌
model.train()

index = 0

# for j, loss_func in lossList:
#     for k, optim in optimizerList:
for j, loss_func in enumerate(lossList):
    for k, optim in enumerate(optimizerList):
        criterion = loss_func
        optimizer = optim

        print("criterion : ", criterion)
        print("optimizer : ", optimizer)
        print()

        # ----------------------------
        # epoch training
        loss = []
        loss_history = []
        model.apply(init_weights)
        for i in range(30):
            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 입력값 생성하고
            a = torch.randn(12, 3, 100, 100).to(device)

            # 모델에 넣은다음
            result = model(a)

            # 결과와 동일한 shape을 가진 Ground-Truth 를 읽어서
            # target  = torch.randn_like(result)

            # 타겟값을 1로 바꾸어서 네트워크가 무조건 1만 출력하도록 만든다.
            target = torch.ones_like(result)

            '''
            print(target.dtype)
            target = target.long()
            target = target.to(device)
            '''

            # 네트워크값과의 차이를 비교
            # loss = criterion(result, target).to(device)

            loss = criterion(result, target).to(device)

            # loss_graph.append(loss.item())
            # loss_graph[index].append(loss.item())  # 현재 조합에 대한 loss 값을 저장
            loss_history.append(loss.item())
            loss_graph[index].append(loss.item())
            iter_graph.append(i)

            # =============================
            # loss는 텐서이므로 item()
            print("epoch: {} loss:{} ".format(i, loss.item()))

            # loss diff값을 뒤로 보내서 grad에 저장하고
            loss.backward()

            # 저장된 grad값을 기준으로 activation func을 적용한다.
            optimizer.step()

        index += 1


    # loss_graph.append(loss_history)
    # index = j * len(optimizerList) + k
    # loss_graph[j * len(optimizerList) + k] = loss_history


#print("=========== 학습된 파라미터만 저장 ==============")
#torch.save(model.state_dict(), 'trained_model.pt')


#모델이 바뀌었으므로 모델 전체를 저장
#print("=========== 전체모델 저장 : VGG 처럼 모델 전체 저장==============")
#torch.save(model, 'trained_model_all.pt')

#loss 그래프 출력
# plt.plot(iter_graph, loss_graph, '-b', label='loss')
# plt.title('loss graph')
# plt.ylabel('loss')
# plt.legend(loc='upper left')
# plt.savefig("result6.png")  # should before show method
# plt.show()

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
index = 0
for j in range(len(lossList)):
    for k in range(len(optimizerList)):
        ax = axes[j][k]
        ax.plot(range(len(loss_graph[index])), loss_graph[index], '-b')
        ax.set_title(f'Loss: {type(lossList[j]).__name__}, Optimizer: {type(optimizerList[k]).__name__}')
        # ax.set_title(f'Loss {j+1}, Optimizer {k+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')

        index += 1
        if index >= len(loss_graph):
            break

plt.tight_layout()  # 서브플롯 간 간격 조정
plt.savefig("loss X optimizer combine.png")
plt.show()