# import library
import copy  # copy object
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Resize
import torchvision.datasets as Datasets
import time
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.offline as pyo
import random
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from facenet_pytorch import MTCNN

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import base64
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define vgg model
class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

# define model type
vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M']
vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
                512, 512, 'M', 512, 512, 512, 512, 'M']


# Define VGG-layer

def get_vgg_layers(config, batch_norm):
    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':  # adapt Maxpooling for 'M'
            layers += [nn.MaxPool2d(kernel_size=2)]

        else:  # adapt Conv2d for numbers
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:  # batch normalization
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                # batch normalization + ReLU
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)  # 네트워크 모든 계층 반환

# vgg11_layers = get_vgg_layers(vgg11_config, batch_norm=True)
vgg16_layers = get_vgg_layers(vgg16_config, batch_norm=True)
# print(vgg11_layers)

output_dim = 11 # class numb
model = VGG(vgg16_layers, output_dim)
# print(model)

# use pretrained VGG model
import torchvision.models as models
pretrained_model = models.vgg11_bn(pretrained=True).to(device)
# print(pretrained_model)

# image preprocess
train_transforms = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomRotation(5),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                    ])

test_transforms = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229,0.224, 0.225])
])

train_path = './train'
test_path = './test'

train_dataset = torchvision.datasets.ImageFolder(
                    train_path,
                    transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(
                    test_path,
                    transform=test_transforms)

print(len(train_dataset), len(test_dataset))

# split dataset
valid_size = 0.9
n_train_examples = int(len(train_dataset) * valid_size)
n_valid_examples = len(train_dataset) - n_train_examples

train_data, valid_data = data.random_split(train_dataset,
                                        [n_train_examples, n_valid_examples])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

print(f'Number of training examples : {len(train_data)}')
print(f'Number of validation examples : {len(valid_data)}')
print(f'Number of testing examples : {len(test_dataset)}')

batch_size = 32

train_iterator = data.DataLoader(train_data,
                                shuffle=True,
                                batch_size=batch_size)

valid_iterator = data.DataLoader(valid_data,
                                batch_size=batch_size)

test_iterator= data.DataLoader(test_dataset,
                              batch_size=batch_size)


# optimizer = optim.Adam(model.parameters(), lr=1e-7)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

def calculate_accuracy(y_pred, y) :
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    highest_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred, _ = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    highest_acc = 0  # highest_acc 변수 초기화

    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()


        return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start, end) :
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# train model
epochs = 100

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

def train_model(model, train_iterator, valid_iterator, optimizer, criterion, device, epochs):
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        print("-----------epochs----------", epoch)
        start = time.monotonic()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './data/VGG-model.pt')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        end = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start, end)

        print(f'Epoch : {epoch + 1:02} | Epoch Time : {epoch_mins}m {epoch_secs}s')
        print(f'\t Train Loss : {train_loss:.3f} | Train Acc : {train_acc * 100:.2f}%')
        print(f'\t Valid Loss : {valid_loss:.3f} | Valid Acc : {valid_acc * 100:.2f}%')

    # 테스트 데이터셋 성능 측정
    model.load_state_dict(torch.load('./data/VGG-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    print(f'Test Loss : {test_loss:.3f} | Test Acc : {test_acc * 100:.2f}%')



correct_examples = []
# 모델 예측 확인
def get_predictions(model, iterator):
    model.eval()
    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred, _ = model(x)
            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


images, labels, probs = get_predictions(model, test_iterator)
pred_labels = torch.argmax(probs, 1)
corrects = torch.eq(labels, pred_labels)

for image, label, prob, correct in zip(images, labels, probs, corrects):
    if correct:
        correct_examples.append((image, label, prob))

correct_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)


def normalize_image(image):  # 이미지를 출력하기 위해 전처리
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_most_correct(correct, classes, n_images, normalize=True):  # 이미지 출력용 함수
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize=(25, 20))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image, true_label, probs = correct[i]
        image = image.permute(1, 2, 0)
        true_prob = probs[true_label]
        correct_prob, correct_label = torch.max(probs, dim=0)
        true_class = classes[true_label]
        correct_class = classes[correct_label]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'True label : {true_class} ({true_prob:.3f})\n'
                     f'Pred label : {correct_class} ({correct_prob:.3f})')
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)


def predict_image(check_image_path, model_path, device):
    # 이미지를 전처리하는 변환(transform) 정의
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # MTCNN 초기화
    mtcnn = MTCNN()

    check_image = Image.open(check_image_path)
    check_image = check_image.convert("RGB")

    # check_image = Image.open(io.BytesIO(base64.b64encode(image_data)))
    # check_image = check_image.convert("RGB")

    check_input_tensor = preprocess(check_image)
    check_input_batch = check_input_tensor.unsqueeze(0)
    check_input_batch = check_input_batch.to(device)

    check_input_features = model.features(check_input_batch)
    check_input_features = check_input_features.view(check_input_features.size(0), -1)

    # 가장 높은 유사도를 저장하는 변수 초기화
    max_similarity = -1
    most_similar_princess = ""
    most_similar_image = None

    class_names = test_dataset.classes
    class_list = os.listdir("./test")
    image_list = []

    # 각 클래스 폴더에서 이미지 선택하여 리스트에 추가
    for class_name in class_list:
        class_path = os.path.join("./test", class_name)
        image_files = os.listdir(class_path)
        image_file = random.choice(image_files)  # 랜덤으로 이미지 선택
        image_path = os.path.join(class_path, image_file)
        image_list.append(image_path)

        # 이미지를 불러온다고 가정
        image = Image.open(image_path)
        image = image.convert("RGB")

        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            for box in boxes:
                x, y, w, h = box
                face_image = image.crop((x, y, x + w, y + h))

                input_tensor = preprocess(image)
                input_batch = input_tensor.unsqueeze(0)
                input_batch = input_batch.to(device)

                target_features = model.features(input_batch)
                target_features = target_features.view(target_features.size(0), -1)

                # 유사도 계산
                similarity = torch.cosine_similarity(check_input_features, target_features).item()

                # 유사도가 현재까지의 최대 유사도보다 높으면 최대 유사도와 클래스를 갱신
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_princess = class_name
                    most_similar_image = image.copy()

                print("Image:", image_path)
                print("Similarity:", similarity)

    print("!!!!Most similar princess!!!!:", most_similar_princess)
    return most_similar_image, most_similar_princess


app = dash.Dash(__name__)

# check_image_path = './rlagPtn.jpg'
model_path = './data/VGG-model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# predict_image(check_image_path, model_path, device)

app.layout = html.Div([
    html.H1("Most Similar Princess"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='uploaded-image-container', style={'text-align': 'center'}),
    html.Button('Predict', id='predict-button', n_clicks=0, style={'fontSize': '20px', 'marginTop': '20px'}),
    html.Div(id='output-prediction', style={'text-align': 'center'})
])


@app.callback(Output('uploaded-image-container', 'children'),
              Input('upload-image', 'contents'))
def update_uploaded_image(contents):
    if contents is not None:
        return html.Img(src=contents, style={'width': '300px'})


@app.callback(Output('output-prediction', 'children'),
              Input('predict-button', 'n_clicks'),
              State('upload-image', 'contents'))
def update_prediction(n_clicks, contents):
    if n_clicks > 0 and contents is not None:
        image_data = contents.split(',')[1]  # 이미지 데이터 추출
        image_data = base64.b64decode(image_data)  # base64 디코딩
        check_image_path = io.BytesIO(image_data)

        most_similar_image, most_similar_princess = predict_image(check_image_path, model_path, device)

        # Most similar image를 base64로 인코딩
        similar_image_data = io.BytesIO()
        most_similar_image.save(similar_image_data, format='JPEG')
        similar_image_base64 = base64.b64encode(similar_image_data.getvalue()).decode("ascii")

        return [
            html.H2("Most Similar Princess"),
            html.H3(most_similar_princess),
            html.H2("Most Similar Image"),
            html.Img(src="data:image/jpeg;base64,{}".format(similar_image_base64))
        ]


if __name__ == '__main__':
    # app.run_server(debug=True)
    app.server.run()

# train_model(model, train_iterator, valid_iterator, optimizer, criterion, device, epochs)

# classes = test_dataset.classes
# n_images = 5
# plot_most_correct(correct_examples, classes, n_images)
# plt.show()


# #Create subplots
# fig = sp.make_subplots(rows=2, cols=1, subplot_titles=("Loss", "Accuracy"))
#
# #Add loss trace to subplot 1
# fig.add_trace(go.Scatter(x=list(range(epochs)), y=train_losses, mode='lines', name='Training Loss'), row=1, col=1)
# fig.add_trace(go.Scatter(x=list(range(epochs)), y=valid_losses, mode='lines', name='Validation Loss'), row=1, col=1)
#
# # Add accuracy trace to subplot 2
# fig.add_trace(go.Scatter(x=list(range(epochs)), y=train_accs, mode='lines', name='Training Accuracy'), row=2, col=1)
# fig.add_trace(go.Scatter(x=list(range(epochs)), y=valid_accs, mode='lines', name='Validation Accuracy'), row=2, col=1)
#
# # Update layout
# fig.update_layout(height=800, width=800, showlegend=False)
#
# # Plot the figure
# pyo.plot(fig, filename='loss_accuracy_plot.html', auto_open=True)

