
import base64
import binascii
import json
import logging

import cv2
import numpy as np

from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .models import Estudiante, SesionAsistencia, Asistencia
from .reconocedor import reconocer_rostro


logger = logging.getLogger(__name__)

# Detector facial de OpenCV
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

# Coincidencias consecutivas necesarias
COINCIDENCIAS_NECESARIAS = 4


# ==========================================
# 1. OBTENER RESUMEN DE ASISTENCIAS
# ==========================================

def obtener_resumen():
    hoy = timezone.localdate()

    total = Estudiante.objects.filter(
        activo=True
    ).count()

    presentes = Asistencia.objects.filter(
        sesion__fecha=hoy,
        estudiante__activo=True
    ).count()

    return {
        "total": total,
        "presentes": presentes,
        "pendientes": max(total - presentes, 0)
    }


# ==========================================
# 2. OBTENER HISTORIAL DEL DÍA
# ==========================================

def obtener_historial():
    hoy = timezone.localdate()

    registros = (
        Asistencia.objects
        .filter(
            sesion__fecha=hoy,
            estudiante__activo=True
        )
        .select_related("estudiante")
        .order_by("-fecha_hora", "-id")[:10]
    )

    historial = []

    for registro in registros:
        historial.append({
            "codigo": registro.estudiante.codigo,
            "nombre": registro.estudiante.nombre,
            "hora": timezone.localtime(
                registro.fecha_hora
            ).strftime("%H:%M:%S"),
            "estado": "Presente"
        })

    return historial


# ==========================================
# 3. INTERFAZ PRINCIPAL
# ==========================================

def inicio(request):
    contexto = {
        "resumen": obtener_resumen(),
        "historial": obtener_historial()
    }

    return render(
        request,
        "asistencia/inicio.html",
        contexto
    )


# ==========================================
# 4. REGISTRAR ASISTENCIA AUTOMÁTICA
# ==========================================

def registrar_asistencia(request, rostros):

    # Solo procesar si aparece un único
    # rostro con identificación preliminar
    if len(rostros) != 1 or not rostros[0]["identificado"]:

        request.session.pop("racha_facial", None)

        return {
            "estado": "SIN_IDENTIFICAR",
            "mensaje": "Esperando un rostro reconocido."
        }

    rostro = rostros[0]
    estudiante_id = rostro["estudiante_id"]

    ahora = timezone.now()
    hoy = timezone.localdate()

    # Recuperar coincidencias anteriores
    racha = request.session.get(
        "racha_facial",
        {}
    )

    intervalo = (
        ahora.timestamp()
        - racha.get("instante", 0)
    )

    misma_persona = (
        racha.get("estudiante_id") == estudiante_id
        and racha.get("fecha") == str(hoy)
        and 0 <= intervalo < 3
    )

    if misma_persona:
        contador = min(
            racha.get("contador", 0) + 1,
            COINCIDENCIAS_NECESARIAS
        )
    else:
        contador = 1

    # Guardar seguimiento de coincidencias
    request.session["racha_facial"] = {
        "estudiante_id": estudiante_id,
        "fecha": str(hoy),
        "instante": ahora.timestamp(),
        "contador": contador
    }

    # Esperar las coincidencias requeridas
    if contador < COINCIDENCIAS_NECESARIAS:

        return {
            "estado": "VERIFICANDO",
            "mensaje": (
                f"Verificando identidad: "
                f"{contador}/{COINCIDENCIAS_NECESARIAS}"
            )
        }

    # Obtener sesión de asistencia del día
    sesion, _ = SesionAsistencia.objects.get_or_create(
        fecha=hoy,
        defaults={
            "estado": "ACTIVA"
        }
    )

    if sesion.estado != "ACTIVA":
        return {
            "estado": "CERRADA",
            "mensaje": "La sesión de asistencia está cerrada."
        }

    # Evitar duplicados mediante la base de datos
    asistencia, creado = Asistencia.objects.get_or_create(
        estudiante_id=estudiante_id,
        sesion=sesion,
        defaults={
            "metodo_verificacion": "FACIAL"
        }
    )

    hora = timezone.localtime(
        asistencia.fecha_hora
    ).strftime("%H:%M:%S")

    if creado:
        return {
            "estado": "REGISTRADA",
            "mensaje": (
                f"Asistencia registrada: "
                f"{rostro['nombre']} - {hora}"
            )
        }

    return {
        "estado": "YA_REGISTRADA",
        "mensaje": (
            f"{rostro['nombre']} ya tiene "
            f"su asistencia registrada."
        )
    }


# ==========================================
# 5. PROCESAR CÁMARA CON OPENCV
# ==========================================

@require_POST
def detectar_rostros(request):

    try:
        # Limitar tamaño de solicitud
        if len(request.body) > 6000000:
            return JsonResponse(
                {"error": "La imagen es demasiado grande."},
                status=413
            )

        datos = json.loads(request.body)

        if not isinstance(datos, dict):
            return JsonResponse(
                {"error": "Solicitud no válida."},
                status=400
            )

        imagen_base64 = datos.get("imagen", "")

        if not isinstance(imagen_base64, str):
            return JsonResponse(
                {"error": "Imagen no válida."},
                status=400
            )

        if not imagen_base64.startswith(
            "data:image/jpeg;base64,"
        ):
            return JsonResponse(
                {"error": "Formato de imagen incorrecto."},
                status=400
            )

        imagen_base64 = imagen_base64.split(",", 1)[1]

        imagen_bytes = base64.b64decode(
            imagen_base64,
            validate=True
        )

        arreglo = np.frombuffer(
            imagen_bytes,
            dtype=np.uint8
        )

        imagen = cv2.imdecode(
            arreglo,
            cv2.IMREAD_COLOR
        )

        if imagen is None:
            return JsonResponse(
                {"error": "No se pudo leer la imagen."},
                status=400
            )

        if detector.empty():
            return JsonResponse(
                {"error": "Detector facial no disponible."},
                status=503
            )

        # Convertir a escala de grises
        gris = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2GRAY
        )

        # Detectar rostros
        detecciones = detector.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        resultados = []

        for x, y, w, h in detecciones:

            rostro_gris = gris[
                y:y+h,
                x:x+w
            ]

            reconocimiento = reconocer_rostro(
                rostro_gris
            )

            resultados.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "estudiante_id": reconocimiento["estudiante_id"],
                "nombre": reconocimiento["nombre"],
                "identificado": reconocimiento["identificado"],
                "distancia": reconocimiento["distancia"]
            })

        # Registrar coincidencias consistentes
        resultado_asistencia = registrar_asistencia(
            request,
            resultados
        )

        # Devolver todos los datos al HTML
        return JsonResponse({
            "rostros": resultados,
            "ancho": imagen.shape[1],
            "alto": imagen.shape[0],
            "asistencia": resultado_asistencia,
            "resumen": obtener_resumen(),
            "historial": obtener_historial()
        })

    except (
        json.JSONDecodeError,
        ValueError,
        binascii.Error
    ):
        return JsonResponse(
            {"error": "Datos de imagen incorrectos."},
            status=400
        )

    except FileNotFoundError:
        return JsonResponse(
            {"error": "No se encontró el modelo facial."},
            status=503
        )

    except cv2.error:
        logger.exception("Error de OpenCV")

        return JsonResponse(
            {"error": "Error al procesar el rostro."},
            status=500
        )

    except Exception:
        logger.exception("Error interno de VirtuAsist")

        return JsonResponse(
            {"error": "Error interno del sistema."},
            status=500
        )
