from django import forms
from .models import ModelFile
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

class ImageForm(forms.ModelForm):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fields['image'].widget.attrs['id'] = 'image-input'
  
  class Meta:
    model = ModelFile
    fields = ('image',)
    
class LoginForm(AuthenticationForm):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for field in self.fields.values():
      field.widget.attrs['class'] = 'form-control'
      field.widget.attrs['placeholder'] = field.label
    self.fields['username'].widget.attrs['id'] = 'username'
    self.fields['password'].widget.attrs['id'] = 'password'
      
class SignUpForm(UserCreationForm):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for field in self.fields.values():
      field.widget.attrs['class'] = 'form-control'
      field.widget.attrs['placeholder'] = field.label
    self.fields['username'].widget.attrs['id'] = 'username'
    self.fields['password1'].widget.attrs['id'] = 'password1'
    self.fields['password2'].widget.attrs['id'] = 'password2'
      
  class Meta:
    model = User
    fields = ('username', 'password1', 'password2',)