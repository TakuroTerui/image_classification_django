from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageForm, LoginForm, SignUpForm
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required

import joblib
import torch
import numpy as np
from .models import ModelFile
import torchvision
from torchvision import transforms, datasets
from torchvision.models import resnet18
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ネットワークの定義
class Net(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.feature = resnet18(pretrained=True)
    self.fc = nn.Linear(1000, 4)

  def forward(self, x):
    h = self.feature(x)
    h = self.fc(h)
    return h

  def training_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t)
    self.log('train_loss', loss, on_step=False, on_epoch=True)
    self.log('train_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t)
    self.log('val_loss', loss, on_step=False, on_epoch=True)
    self.log('val_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, t = batch
    y = self(x)
    loss = F.cross_entropy(y, t)
    self.log('test_loss', loss, on_step=False, on_epoch=True)
    self.log('test_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    return optimizer

# モデルの読み込み
loaded_model = Net().cpu().eval()
loaded_model.load_state_dict(torch.load('model/image.pt'))

# 正解ラベル
label = {
  0 : 'ゴリラ',
  1 : 'ゾウ',
  2 : 'パンダ',
  3 : 'ホッキョクグマ',
}

@login_required
def hellofunction(request):
  return HttpResponse('Hello World!')

@login_required
def image_upload(request):
  if request.method == 'POST':
    form = ImageForm(request.POST, request.FILES)
    if form.is_valid():
      form.save()
      model_file = ModelFile.objects.order_by('id').reverse()[0]
      img_url = '/media/{}'.format(model_file.image)
      result_list = inference('media/{}'.format(model_file.image))
    return render(request, 'send_image_app/classify.html', {'img_url':img_url, 'result_list':result_list})
  else:
      form = ImageForm()
      return render(request, 'send_image_app/index.html', {'form':form})
    
class Login(LoginView):
  form_class = LoginForm
  template_name = 'send_image_app/login.html'
  
class Logout(LogoutView):
  template_name = 'send_image_app/base.html'
  
# サインアップ
def signup(request):
  if request.method == 'POST':
    form = SignUpForm(request.POST)
    if form.is_valid():
      form.save()
      username = form.cleaned_data.get('username')
      password = form.cleaned_data.get('password')
      new_user = authenticate(username=username, password=password)
      if new_user is not None:
        login(request, new_user)
        return redirect('index')
  else:
    form = SignUpForm()
    return render(request, 'send_image_app/signup.html', {'form':form})
  
# 推論処理
def inference(img_url):
  model_file = ModelFile.objects.order_by('id').reverse()[0]
  img = Image.open(img_url)
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  img = transform(img)
  # 予測値の算出
  y = loaded_model(img.unsqueeze(0))
  # 確率に変換
  y = F.softmax(y)
  proba = torch.max(y).item() * 100
  model_file.proba = proba
  # 予測ラベル
  y = torch.argmax(y)
  model_file.result = y.item()
  result_list = {
    'proba' : proba,
    'animal_name' : label[y.item()],
  }
  return result_list
