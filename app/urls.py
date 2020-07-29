from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name = "upload-and-process"),
    path('result/', views.process, name = "show-result"),
]
