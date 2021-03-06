from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import *
from .SearchObject import *
from .models import *
import csv
from django.core.files.storage import FileSystemStorage

s1 = SearchObject('media/images', 'media/videos')

# queryset = Post.objects.all()


# def query_to_csv(queryset, filename='items.csv', **override):
#     field_names = [field.name for field in queryset.model._meta.fields]

#     def field_value(row, field_name):
#         if field_name in override.keys():
#             return override[field_name]
#         else:
#             return row[field_name]
#     with open(filename, 'w+', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter=',')
#         writer.writerow(field_names)
#         for row in queryset.values(*field_names):
#             writer.writerow([field_value(row, field) for field in field_names])


# query_to_csv(queryset, filename='data.csv', user=1, group=1)


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
    posts = Post.objects.all().order_by('-created_on')[:1]
    data = s1.process()
    s, y = data.keys(), data.values()
    d = read_data('data.csv')
    key = d['image']
    img_src = 'media/' + key[0]
    processed_img = 'media/new.jpg'
    context = {
        "keys": s,
        "values": y,
        "image": img_src,
        "p_img": processed_img,
        "posts": posts,
    }

    return render(request, 'show.html', context)


def upload_video(request):
    if request.method == 'POST':
        upload_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(upload_file.name, upload_file)
        url = fs.url(name)
        print(url)
    return render(request, 'upload_video.html')


def upload2(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            s1.process_video()
            return redirect('upload')
    else:
        form = VideoForm()
    return render(request, 'upload2.html', {'form': form})
