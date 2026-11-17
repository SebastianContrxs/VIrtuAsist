import csv

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from asistencia.models import Estudiante


class Command(BaseCommand):
    help = "Registra o actualiza estudiantes desde estudiantes.csv."

    def handle(self, *args, **options):
        archivo = settings.BASE_DIR / "estudiantes.csv"
        filas = []
        codigos = set()
        try:
            with archivo.open(encoding="utf-8-sig", newline="") as entrada:
                lector = csv.DictReader(entrada)
                if lector.fieldnames != ["codigo", "nombre"]:
                    raise CommandError("El CSV debe comenzar con codigo,nombre.")
                for fila in lector:
                    codigo = (fila.get("codigo") or "").strip()
                    nombre = (fila.get("nombre") or "").strip()
                    if None in fila or not codigo or not nombre:
                        raise CommandError(
                            f"Fila {lector.line_num}: usa CODIGO,Nombre del estudiante."
                        )
                    if len(codigo) > 20 or len(nombre) > 120:
                        raise CommandError(f"Fila {lector.line_num}: codigo o nombre demasiado largo.")
                    if codigo in {".", ".."} or any(c in codigo for c in '/\\:*?"<>|'):
                        raise CommandError(f"Fila {lector.line_num}: codigo no valido para una carpeta.")
                    if codigo in codigos:
                        raise CommandError(f"Fila {lector.line_num}: codigo duplicado: {codigo}.")
                    codigos.add(codigo)
                    filas.append((codigo, nombre))
        except (OSError, UnicodeError, csv.Error) as error:
            raise CommandError(f"No se pudo leer estudiantes.csv: {error}") from error

        with transaction.atomic():
            for codigo, nombre in filas:
                Estudiante.objects.update_or_create(
                    codigo=codigo,
                    defaults={"nombre": nombre, "activo": True},
                )

        self.stdout.write(self.style.SUCCESS(f"Estudiantes procesados: {len(filas)}"))
