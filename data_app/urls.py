from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload_file'),
    path('analysis/<int:dataset_id>/', views.analyze_data, name='analysis'),
]