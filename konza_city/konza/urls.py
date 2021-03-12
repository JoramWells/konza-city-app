from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.home, name='home'),
    path('upload', views.index, name='index'),
    path('show', views.show, name='show'),
    path('upload2', views.upload_video, name='upload'),
    path('upload3', views.upload2, name='upload2')


]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
