import cv2

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

from asistencia.management.commands.generar_modelo_facial import EXTENSIONES
from asistencia.models import Estudiante


class Command(BaseCommand):
    help = "Prepara las tablas, registra estudiantes y actualiza el modelo facial si hay fotos."

    def handle(self, *args, **options):
        for comando in ("preparar_base_datos", "registrar_estudiantes"):
            call_command(comando, stdout=self.stdout, stderr=self.stderr)

        hay_fotos = False
        for estudiante in Estudiante.objects.filter(activo=True):
            carpeta = settings.CARPETA_FOTOGRAFIAS / estudiante.codigo
            carpeta.mkdir(parents=True, exist_ok=True)
            hay_fotos = hay_fotos or any(
                archivo.is_file() and archivo.suffix.lower() in EXTENSIONES
                for archivo in carpeta.iterdir()
            )

        if hay_fotos:
            call_command("generar_modelo_facial", stdout=self.stdout, stderr=self.stderr)
        elif settings.RUTA_MODELO_FACIAL.is_file():
            self.stdout.write(self.style.WARNING(
                "No hay fotografías: se conserva el modelo existente. "
                "Agrega fotos y ejecuta el comando generar_modelo_facial para entrenar con los estudiantes actuales."
            ))
        else:
            raise CommandError(
                "Agrega fotografías en fotografias_estudiantes/CODIGO/ "
                "y ejecuta el comando generar_modelo_facial. No existe un modelo facial."
            )

        try:
            modelo = cv2.face.LBPHFaceRecognizer_create()
            modelo.read(str(settings.RUTA_MODELO_FACIAL))
            etiquetas = {int(valor) for valor in modelo.getLabels().flatten()}
            if not etiquetas:
                raise CommandError("El modelo facial no contiene rostros entrenados.")
        except (cv2.error, AttributeError) as error:
            raise CommandError(f"No se pudo cargar el modelo facial: {error}") from error

        sin_modelo = list(Estudiante.objects.filter(activo=True).exclude(
            id__in=etiquetas
        ).values_list("codigo", flat=True))
        if sin_modelo:
            self.stdout.write(self.style.WARNING(
                "Estudiantes sin muestras en el modelo: " + ", ".join(sin_modelo)
            ))
        self.stdout.write(self.style.SUCCESS("Preparación terminada. Inicia el servidor con manage.py runserver."))
