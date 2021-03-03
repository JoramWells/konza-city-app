from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import *
from .SearchObject import *
from .models import *
import csv


s1 = SearchObject('media/images')

queryset = Post.objects.all()


def query_to_csv(queryset, filename='items.csv', **override):
    field_names = [field.name for field in queryset.model._meta.fields]

    def field_value(row, field_name):
        if field_name in override.keys():
            return override[field_name]
        else:
            return row[field_name]
    with open(filename, 'w+', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter=',')
        writer.writerow(field_names)
        for row in queryset.values(*field_names):
            writer.writerow([field_value(row, field) for field in field_names])


query_to_csv(queryset, filename='data.csv', user=1, group=1)


def index(request):
    if request.method == 'POST':
        form = InputForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()

            return redirect('show')

    else:
        form = Post()
    return render(request, 'index.html', {'form': form})


def success(request):
    return HttpResponse('successfull uploaded')


def show(request):
    data = s1.process()
    context = {
        "url": data
    }

    return render(request, 'show.html', context)
