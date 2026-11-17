
from django.db import models
from django.utils import timezone


# Tabla de estudiantes
class Estudiante(models.Model):
    codigo = models.CharField(max_length=20, unique=True)
    nombre = models.CharField(max_length=120)
    activo = models.BooleanField(default=True)

    def __str__(self):
        return self.nombre


# Tabla de sesiones de asistencia
class SesionAsistencia(models.Model):
    fecha = models.DateField(unique=True)
    hora_inicio = models.DateTimeField(default=timezone.now)
    estado = models.CharField(
        max_length=10,
        choices=[
            ("ACTIVA", "Activa"),
            ("CERRADA", "Cerrada"),
        ],
        default="ACTIVA"
    )

    def __str__(self):
        return f"Sesion del {self.fecha}"


# Tabla de asistencias registradas
class Asistencia(models.Model):
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.CASCADE
    )

    sesion = models.ForeignKey(
        SesionAsistencia,
        on_delete=models.CASCADE
    )

    fecha_hora = models.DateTimeField(default=timezone.now)

    metodo_verificacion = models.CharField(
        max_length=20,
        default="FACIAL"
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["estudiante", "sesion"],
                name="asistencia_unica_por_sesion"
            )
        ]

    def __str__(self):
        return f"{self.estudiante.nombre} - {self.fecha_hora}"
