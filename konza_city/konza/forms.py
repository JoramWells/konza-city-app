from django import forms
from .models import *


class InputForm(forms.ModelForm):
    name = forms.CharField(widget=forms.TextInput(attrs={
                           'class': 'form-control', 'placeholder': 'Enter highway name'}))

    class Meta:
        model = Post
        fields = ['name', 'image']


class VideoForm(forms.ModelForm):
    class Meta:
        model = VideoModel
        fields = ['title', 'video']
