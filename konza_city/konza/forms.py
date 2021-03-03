from django import forms
from .models import *
class InputForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['name', 'image']