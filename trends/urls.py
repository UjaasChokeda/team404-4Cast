from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('about/', views.about, name='about'),
    path('feedback/',views.feedback, name='feedback'),
    path('faq/',views.faq, name='faq'),
    path('trending',views.trending, name='trending'),
    path('save_feedback/', views.save_feedback, name='save_feedback'),
    path('save_to_text_file/', views.save_to_text_file, name='save_to_text_file'),
]
