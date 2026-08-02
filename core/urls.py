from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),  # ✅ THIS FIXES 404
    path('video_feed/', views.video_feed, name='video_feed'),
    path('get-posture-data/', views.get_posture_data, name='get_posture_data'),

]