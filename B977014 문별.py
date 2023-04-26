#Pretrained NN을 튜닝한다.

import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary

import plotly.express as px
import pandas as pd


#cuda로 보낸다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('========VGG 테스트 =========')
print("========입력데이터 생성 [batch, color, image x, image y]=========")
#이미지 사이즈를 어떻게 잡아도 vgg는 다 소화한다.


#==========================================
#1) 모델 생성
model = models.vgg16(pretrained=True).to(device)

print(model)
print('========= Summary로 보기 =========')
#Summary 때문에 cuda, cpu 맞추어야 함
#뒤에 값이 들어갔을 때 내부 변환 상황을 보여줌
#adaptive average pool이 중간에서 최종값을 바꿔주고 있음
summary(model, (3, 100, 100))

print("========model weight 값 측정=========")
'''
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
'''

#2) loss function
#꼭 아래와 같이 2단계, 클래스 선언 후, 사용
criterion = nn.MSELoss()

#3) activation function
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#모델이 학습 모드라고 알려줌
model.train()

result_list = []
min_loss = 0

#----------------------------
#epoch training
for i in range (10):

    #옵티마이저 초기화
    optimizer.zero_grad()

    #입력값 생성하고
    a = torch.randn(12,3,100,100).to(device)

    #모델에 넣은다음
    result = model(a)

    #결과와 동일한 shape을 가진 Ground-Truth 를 읽어서
    target  = torch.randn_like(result)

    #네트워크값과의 차이를 비교
    loss = criterion(result, target).to(device)
    result_list.append(loss.item())

    #=============================
    #loss는 텐서이므로 item()
    print("epoch: {} loss:{} ".format(i, loss.item()))

    # 첫 번째 값을 min_loss라고 지정
    if i==0:
        min_loss = loss.item()
        print("i==0 일 때 min_loss", min_loss)
    #다음에 검사하는 값이 min_loss보다 작으면 min_loss 교체
    elif loss.item() < min_loss:
        min_loss = loss.item()
        print("loss 값이 min_loss보다 작을 때 교체한 min_loss값", min_loss)
    #다음에 들어오는 값이 min_loss보다 크면 loss값이 증가한 것이므로
    elif loss.item() > min_loss:
        print("loss값이 min_loss보다 클 때의 loss값", loss.item())
        result_list.pop()
        break

    #loss diff값을 뒤로 보내서 grad에 저장하고
    loss.backward()

    #저장된 grad값을 기준으로 activation func을 적용한다.
    optimizer.step()


df = pd.DataFrame(dict(
    x = range(len(result_list)),
    y = result_list
))

if len(result_list) > 1:
    fig = px.line(df, x="x", y="y", title = "Unsorted Input")
    fig.show()

# print(result_list)

print("=========== 학습된 파라미터만 저장 ==============")
torch.save(model.state_dict(), 'trained_model.pt')

print("=========== 전체모델 저장 : VGG 처럼 모델 전체 저장==============")
torch.save(model, 'trained_model_all.pt')