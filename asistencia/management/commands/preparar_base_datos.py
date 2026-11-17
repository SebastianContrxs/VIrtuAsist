from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Crea o actualiza las tablas de la base de datos."

    def handle(self, *args, **options):
        call_command("migrate", interactive=False, stdout=self.stdout, stderr=self.stderr)
