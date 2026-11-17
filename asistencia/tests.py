import base64
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from .models import Asistencia, Estudiante


class RegistroEstudiantesTests(TestCase):
    def importar(self, contenido):
        with TemporaryDirectory() as carpeta:
            ruta = Path(carpeta)
            (ruta / "estudiantes.csv").write_text(contenido, encoding="utf-8")
            with override_settings(BASE_DIR=ruta):
                call_command("registrar_estudiantes", stdout=StringIO())

    def test_actualizar_conserva_id_y_no_duplica(self):
        estudiante = Estudiante.objects.create(codigo="EST001", nombre="Anterior")
        self.importar("codigo,nombre\nEST001,Actualizado\nEST002,Nuevo\n")
        self.importar("codigo,nombre\nEST001,Actualizado\nEST002,Nuevo\n")
        estudiante.refresh_from_db()
        self.assertEqual(estudiante.nombre, "Actualizado")
        self.assertEqual(Estudiante.objects.count(), 2)

    def test_fila_incompleta_no_importa_datos_parciales(self):
        with self.assertRaisesMessage(CommandError, "Fila 3"):
            self.importar("codigo,nombre\nEST001,Correcto\nEST002SinComa\n")
        self.assertFalse(Estudiante.objects.exists())

    def test_codigo_no_puede_salir_de_carpeta_fotografias(self):
        with self.assertRaises(CommandError):
            self.importar("codigo,nombre\n../fuera,Nombre\n")
        self.assertFalse(Estudiante.objects.exists())


class ModeloFacialTests(TestCase):
    def test_sin_fotos_conserva_modelo_anterior_en_ruta_nueva(self):
        with TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            fotos = raiz / "fotografias_estudiantes"
            fotos.mkdir()
            modelo = raiz / "reconocimiento_facial" / "modelo_asistencia.yml"
            modelo.parent.mkdir()
            modelo.write_bytes(b"modelo anterior")
            with override_settings(CARPETA_FOTOGRAFIAS=fotos, RUTA_MODELO_FACIAL=modelo):
                with self.assertRaisesMessage(CommandError, "No se encontraron fotografías válidas"):
                    call_command("generar_modelo_facial", stdout=StringIO())
            self.assertEqual(modelo.read_bytes(), b"modelo anterior")


class InterfazAsistenciaTests(TestCase):
    def test_inicio_y_solicitud_invalida(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/detectar/").status_code, 405)
        respuesta = self.client.post("/detectar/", data={"imagen": "invalida"}, content_type="application/json")
        self.assertEqual(respuesta.status_code, 400)

    def test_imagen_sin_rostros(self):
        _, imagen = cv2.imencode(".jpg", np.zeros((160, 160, 3), dtype=np.uint8))
        respuesta = self.client.post("/detectar/", data={
            "imagen": "data:image/jpeg;base64," + base64.b64encode(imagen).decode()
        }, content_type="application/json")
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(respuesta.json()["rostros"], [])
        self.assertEqual(respuesta.json()["asistencia"]["estado"], "SIN_IDENTIFICAR")

    def test_cuatro_coincidencias_registran_una_sola_asistencia(self):
        estudiante = Estudiante.objects.create(codigo="PRUEBA", nombre="Prueba")
        _, imagen = cv2.imencode(".jpg", np.zeros((160, 160, 3), dtype=np.uint8))
        datos = {"imagen": "data:image/jpeg;base64," + base64.b64encode(imagen).decode()}
        reconocimiento = {
            "estudiante_id": estudiante.id, "nombre": estudiante.nombre,
            "identificado": True, "distancia": 10.0,
        }
        with patch("asistencia.views.detector") as detector, patch(
            "asistencia.views.reconocer_rostro", return_value=reconocimiento
        ):
            detector.empty.return_value = False
            detector.detectMultiScale.return_value = [(0, 0, 160, 160)]
            estados = [
                self.client.post("/detectar/", data=datos, content_type="application/json").json()["asistencia"]["estado"]
                for _ in range(5)
            ]
        self.assertEqual(estados, ["VERIFICANDO"] * 3 + ["REGISTRADA", "YA_REGISTRADA"])
        self.assertEqual(Asistencia.objects.count(), 1)
