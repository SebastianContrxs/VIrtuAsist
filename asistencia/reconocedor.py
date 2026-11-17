
from functools import lru_cache

import cv2

from django.conf import settings
from .models import Estudiante


# ==========================================
# CONFIGURACIÓN DEL RECONOCIMIENTO FACIAL
# ==========================================

# Umbral provisional del reconocimiento.
# Una distancia menor indica mayor similitud.
UMBRAL_RECONOCIMIENTO = 60.0


# ==========================================
# CARGAR MODELO FACIAL ENTRENADO
# ==========================================

@lru_cache(maxsize=2)
def cargar_modelo(ruta, modificacion):

    reconocedor = cv2.face.LBPHFaceRecognizer_create()

    reconocedor.read(ruta)

    return reconocedor


# ==========================================
# RECONOCER EL ROSTRO
# ==========================================

def reconocer_rostro(rostro_gris):

    # Ubicación del modelo entrenado
    ruta_modelo = settings.RUTA_MODELO_FACIAL

    # Verificar que el modelo exista
    if not ruta_modelo.is_file():
        raise FileNotFoundError(
            "No existe el modelo facial entrenado."
        )

    # ======================================
    # PREPROCESAMIENTO
    # ======================================

    # Ajustar tamaño del rostro
    rostro = cv2.resize(
        rostro_gris,
        (160, 160)
    )

    # Mejorar contraste
    rostro = cv2.equalizeHist(rostro)

    # ======================================
    # CARGAR MODELO LBPH
    # ======================================

    reconocedor = cargar_modelo(
        str(ruta_modelo),
        ruta_modelo.stat().st_mtime_ns
    )

    # ======================================
    # REALIZAR PREDICCIÓN
    # ======================================

    etiqueta, distancia = reconocedor.predict(
        rostro
    )

    # ======================================
    # MOSTRAR RESULTADO EN TERMINAL
    # ======================================

    print(
        f"[RECONOCIMIENTO] "
        f"ID detectado: {etiqueta} | "
        f"Distancia: {distancia:.2f} | "
        f"Umbral permitido: {UMBRAL_RECONOCIMIENTO}",
        flush=True
    )

    # ======================================
    # RESULTADO INICIAL
    # ======================================

    resultado = {
        "estudiante_id": None,
        "nombre": "Desconocido",
        "identificado": False,
        "distancia": round(float(distancia), 2)
    }

    # ======================================
    # VERIFICAR UMBRAL
    # ======================================

    if (
        etiqueta < 0
        or distancia > UMBRAL_RECONOCIMIENTO
    ):

        print(
            "[RESULTADO] Rostro rechazado "
            "por distancia o etiqueta inválida.",
            flush=True
        )

        return resultado

    # ======================================
    # BUSCAR ESTUDIANTE EN SQLITE
    # ======================================

    estudiante = Estudiante.objects.filter(
        id=etiqueta,
        activo=True
    ).first()

    if estudiante is None:

        print(
            "[RESULTADO] No se encontró "
            "un estudiante activo con ese ID.",
            flush=True
        )

        return resultado

    # ======================================
    # POSIBLE COINCIDENCIA
    # ======================================

    resultado["estudiante_id"] = estudiante.id
    resultado["nombre"] = estudiante.nombre
    resultado["identificado"] = True

    print(
        "[RESULTADO] Coincidencia aceptada "
        "por el umbral provisional.",
        flush=True
    )

    return resultado
