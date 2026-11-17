
from django.urls import path
from . import views

urlpatterns = [
    path("", views.inicio, name="inicio"),
    path(
        "detectar/",
        views.detectar_rostros,
        name="detectar_rostros"
    ),
]
