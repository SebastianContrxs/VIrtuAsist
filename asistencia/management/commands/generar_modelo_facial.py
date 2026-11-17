

import cv2
import numpy as np

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from asistencia.models import Estudiante


# ==========================================
# CONFIGURACIÓN
# ==========================================

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}

TAMANO_ROSTRO = (160, 160)

MAX_DIMENSION_IMAGEN = 1600


# ==========================================
# CARGAR DETECTORES FACIALES
# ==========================================

def cargar_detectores():

    nombres = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml"
    ]

    detectores = []

    for nombre in nombres:

        ruta = cv2.data.haarcascades + nombre

        detector = cv2.CascadeClassifier(ruta)

        if not detector.empty():
            detectores.append(detector)

    if not detectores:
        raise CommandError(
            "No se pudieron cargar los detectores faciales."
        )

    return detectores


# ==========================================
# LEER IMAGEN
# ==========================================

def leer_imagen(ruta):

    try:
        datos = np.fromfile(str(ruta), dtype=np.uint8)

        imagen = cv2.imdecode(
            datos,
            cv2.IMREAD_COLOR
        )

        return imagen

    except (OSError, ValueError, cv2.error):
        return None


# ==========================================
# AJUSTAR TAMAÑO DE IMAGEN
# ==========================================

def ajustar_imagen(imagen):

    alto, ancho = imagen.shape[:2]

    mayor_dimension = max(alto, ancho)

    if mayor_dimension <= MAX_DIMENSION_IMAGEN:
        return imagen

    escala = MAX_DIMENSION_IMAGEN / mayor_dimension

    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)

    return cv2.resize(
        imagen,
        (nuevo_ancho, nuevo_alto),
        interpolation=cv2.INTER_AREA
    )


# ==========================================
# SELECCIONAR ROSTRO PRINCIPAL
# ==========================================

def seleccionar_rostro(detecciones, ancho, alto):

    if len(detecciones) == 0:
        return None

    # Ordenar posibles rostros por tamaño
    rostros = sorted(
        detecciones,
        key=lambda r: int(r[2]) * int(r[3]),
        reverse=True
    )

    principal = rostros[0]

    x, y, w, h = map(int, principal)

    area_principal = w * h

    area_imagen = ancho * alto

    # Descartar regiones demasiado pequeñas
    if area_principal < area_imagen * 0.015:
        return None

    # Si hay varias regiones, comprobar
    # que exista un rostro claramente dominante
    if len(rostros) > 1:

        segundo = rostros[1]

        area_segundo = (
            int(segundo[2]) * int(segundo[3])
        )

        if area_principal < area_segundo * 2:
            return None

    return (x, y, w, h)


# ==========================================
# DETECTAR ROSTRO CON VARIOS INTENTOS
# ==========================================

def detectar_rostro(imagen, detectores):

    imagen = ajustar_imagen(imagen)

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    alto, ancho = gris.shape[:2]

    # Mejorar contraste de la imagen
    ecualizada = cv2.equalizeHist(gris)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    mejorada = clahe.apply(gris)

    versiones = [
        gris,
        ecualizada,
        mejorada
    ]

    # Intentos progresivos de detección
    configuraciones = [
        (1.1, 5),
        (1.1, 4),
        (1.05, 4),
        (1.05, 3)
    ]

    minimo = max(
        30,
        int(min(ancho, alto) * 0.08)
    )

    for detector in detectores:

        for version in versiones:

            for escala, vecinos in configuraciones:

                rostros = detector.detectMultiScale(
                    version,
                    scaleFactor=escala,
                    minNeighbors=vecinos,
                    minSize=(minimo, minimo)
                )

                rostro = seleccionar_rostro(
                    rostros,
                    ancho,
                    alto
                )

                if rostro is None:
                    continue

                x, y, w, h = rostro

                # Recortar desde la imagen original
                rostro_gris = gris[
                    y:y+h,
                    x:x+w
                ]

                if rostro_gris.size == 0:
                    continue

                # Mismo procesamiento que utilizará
                # el reconocimiento desde la cámara
                rostro_gris = cv2.resize(
                    rostro_gris,
                    TAMANO_ROSTRO
                )

                rostro_gris = cv2.equalizeHist(
                    rostro_gris
                )

                return rostro_gris

    return None


# ==========================================
# COMANDO DE ENTRENAMIENTO
# ==========================================

class Command(BaseCommand):

    help = (
        "Entrena el reconocimiento facial "
        "utilizando fotografías precargadas."
    )

    def handle(self, *args, **options):

        carpeta_fotos = settings.CARPETA_FOTOGRAFIAS
        carpeta_modelos = settings.RUTA_MODELO_FACIAL.parent

        # Comprobar carpeta
        if not carpeta_fotos.is_dir():
            raise CommandError(
                "No existe la carpeta fotografias_estudiantes."
            )

        # Comprobar OpenCV Contrib
        if not (
            hasattr(cv2, "face")
            and hasattr(
                cv2.face,
                "LBPHFaceRecognizer_create"
            )
        ):
            raise CommandError(
                "OpenCV Contrib no está disponible."
            )

        detectores = cargar_detectores()

        imagenes = []
        etiquetas = []

        total_archivos = 0
        total_validas = 0
        total_rechazadas = 0
        estudiantes_entrenados = 0

        estudiantes = Estudiante.objects.filter(
            activo=True
        ).order_by("codigo")

        self.stdout.write(
            "\n===================================="
        )

        self.stdout.write(
            "   ENTRENAMIENTO FACIAL VIRTUASIST"
        )

        self.stdout.write(
            "====================================\n"
        )

        # Recorrer estudiantes
        for estudiante in estudiantes:

            carpeta = (
                carpeta_fotos / estudiante.codigo
            )

            self.stdout.write(
                f"\nEstudiante: {estudiante.nombre}"
            )

            self.stdout.write(
                f"Codigo: {estudiante.codigo}"
            )

            if not carpeta.is_dir():

                self.stdout.write(
                    self.style.WARNING(
                        "No existe la carpeta de fotografías."
                    )
                )

                continue

            fotografias = sorted([
                archivo
                for archivo in carpeta.iterdir()
                if archivo.is_file()
                and archivo.suffix.lower() in EXTENSIONES
            ])

            cantidad_validas = 0

            if not fotografias:

                self.stdout.write(
                    self.style.WARNING(
                        "No se encontraron fotografías."
                    )
                )

                continue

            # Revisar todas las fotografías
            for archivo in fotografias:

                total_archivos += 1

                self.stdout.write(
                    f"\nProcesando: {archivo.name}"
                )

                imagen = leer_imagen(archivo)

                if imagen is None:

                    total_rechazadas += 1

                    self.stdout.write(
                        self.style.WARNING(
                            "Imagen no válida o no legible."
                        )
                    )

                    continue

                rostro = detectar_rostro(
                    imagen,
                    detectores
                )

                if rostro is None:

                    total_rechazadas += 1

                    self.stdout.write(
                        self.style.WARNING(
                            "No se encontró un rostro "
                            "principal suficientemente claro."
                        )
                    )

                    continue

                # Agregar fotografía al entrenamiento
                imagenes.append(rostro)

                # Asociar con ID real de SQLite
                etiquetas.append(estudiante.id)

                cantidad_validas += 1
                total_validas += 1

                self.stdout.write(
                    self.style.SUCCESS(
                        "Fotografía aceptada."
                    )
                )

            if cantidad_validas > 0:
                estudiantes_entrenados += 1

            self.stdout.write(
                self.style.SUCCESS(
                    f"Fotografías válidas para "
                    f"{estudiante.nombre}: {cantidad_validas}"
                )
            )

        # ==================================
        # VERIFICAR IMÁGENES
        # ==================================

        if not imagenes:

            raise CommandError(
                "No se encontraron fotografías válidas. "
                "No se modificó el modelo anterior."
            )

        # ==================================
        # ENTRENAR MODELO LBPH
        # ==================================

        self.stdout.write(
            "\nEntrenando modelo LBPH..."
        )

        reconocedor = (
            cv2.face.LBPHFaceRecognizer_create()
        )

        reconocedor.train(
            imagenes,
            np.array(
                etiquetas,
                dtype=np.int32
            )
        )

        # ==================================
        # GUARDAR MODELO
        # ==================================

        carpeta_modelos.mkdir(
            parents=True,
            exist_ok=True
        )

        ruta_modelo = settings.RUTA_MODELO_FACIAL

        reconocedor.write(
            str(ruta_modelo)
        )

        # ==================================
        # RESULTADOS FINALES
        # ==================================

        self.stdout.write(
            "\n===================================="
        )

        self.stdout.write(
            "      RESULTADOS DEL ENTRENAMIENTO"
        )

        self.stdout.write(
            "===================================="
        )

        self.stdout.write(
            f"Estudiantes entrenados: "
            f"{estudiantes_entrenados}"
        )

        self.stdout.write(
            f"Fotografías encontradas: "
            f"{total_archivos}"
        )

        self.stdout.write(
            f"Fotografías aceptadas: "
            f"{total_validas}"
        )

        self.stdout.write(
            f"Fotografías rechazadas: "
            f"{total_rechazadas}"
        )

        self.stdout.write(
            self.style.SUCCESS(
                "\nEntrenamiento terminado correctamente."
            )
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Modelo guardado en:\n{ruta_modelo}"
            )
        )
