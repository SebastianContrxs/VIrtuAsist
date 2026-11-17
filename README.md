# VirtuAsist

Control de asistencia con Django y reconocimiento facial con OpenCV.

## Ejecutar en este equipo

La instalación ya está realizada en `.venv` y la base de datos está preparada.

1. Abre PowerShell en la carpeta del proyecto y ejecuta:

   ```powershell
   .\.venv\Scripts\python.exe manage.py runserver
   ```

2. Abre **http://127.0.0.1:8000/** en el navegador.
3. Permite el acceso a la cámara y coloca una persona frente a ella.
4. Para detener el servidor, presiona **Ctrl+C** en la terminal.

No necesitas activar el entorno virtual ni escribir los comandos de instalación cada vez.

**Estado de las fotografías:** las carpetas recibidas estaban vacías. Se conserva el modelo facial existente, que puede cargarse correctamente. Para reconocer a los estudiantes actuales con certeza, agrega sus fotografías y vuelve a entrenar. Cambiar el nombre de un estudiante en el CSV no cambia el rostro aprendido. No se ha verificado el reconocimiento con una cámara real.

## Registrar estudiantes y preparar sus rostros

1. Edita `estudiantes.csv`. La primera línea debe ser `codigo,nombre`. Ejemplo:

   ```csv
   codigo,nombre
   EST001,Pepe Tola
   EST002,Bombon Acesino
   EST003,Sebastian Contreras
   ```

2. Coloca varias fotografías claras de cada persona en su carpeta:

   ```text
   fotografias_estudiantes/
     EST001/
       foto1.jpg
       foto2.jpg
     EST002/
       foto1.jpg
     EST003/
       foto1.jpg
   ```

   Las carpetas se crean automáticamente al preparar el sistema. Se admiten JPG, JPEG, PNG y BMP. Usa imágenes con un rostro principal visible y buena iluminación.

3. Ejecuta los siguientes comandos. Al entrenar, incluye fotografías de todos los estudiantes que deban reconocerse: el entrenamiento reemplaza el modelo completo y solo incluye los estudiantes activos con fotos válidas.

   ```powershell
   .\.venv\Scripts\python.exe manage.py registrar_estudiantes
   .\.venv\Scripts\python.exe manage.py generar_modelo_facial
   ```

4. Si el servidor estaba abierto, recarga la página para comenzar una nueva verificación.

El CSV actualiza estudiantes por código y conserva sus identificadores. Repetir la importación no crea duplicados. Los registros ausentes del CSV no se eliminan. Si una fila está incompleta o tiene un código duplicado, se informa el error antes de modificar estudiantes.

Si no hay fotografías, la preparación conserva el modelo existente y muestra un aviso. Si tampoco hay modelo, solicita agregar fotos. Si hay fotos pero ninguna es válida, el entrenamiento falla y conserva el modelo anterior.

## Nombres nuevos

| Antes | Ahora |
| --- | --- |
| `rostros_registrados/` | `fotografias_estudiantes/` |
| `modelos/` | `reconocimiento_facial/` |
| `modelos/rostros_lbph.yml` | `reconocimiento_facial/modelo_asistencia.yml` |
| `cargar_estudiantes` | `registrar_estudiantes` |
| `entrenar_rostros` | `generar_modelo_facial` |
| `migrate` en las instrucciones de uso | `preparar_base_datos` |

`preparar_base_datos` ejecuta internamente las migraciones oficiales de Django. La carpeta interna `asistencia/migrations` conserva su nombre porque Django la utiliza para administrar las tablas. Los comandos antiguos de estudiantes y entrenamiento fueron renombrados; utiliza los nombres nuevos.

## Comandos individuales (PowerShell)

```powershell
# Crear o actualizar las tablas
.\.venv\Scripts\python.exe manage.py preparar_base_datos

# Registrar o actualizar estudiantes desde el CSV
.\.venv\Scripts\python.exe manage.py registrar_estudiantes

# Generar el modelo con las fotografías disponibles
.\.venv\Scripts\python.exe manage.py generar_modelo_facial

# Preparar tablas, estudiantes y modelo en una sola operación
.\.venv\Scripts\python.exe manage.py preparar_sistema

# Iniciar el servidor
.\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

## Instalación en otro equipo Windows

Instala Python 3.13 de 64 bits y habilita su acceso desde la terminal. Copia el proyecto y ejecuta estos comandos. Se necesita internet para descargar las dependencias. No copies `.venv` entre equipos: crea un entorno nuevo en el destino.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe manage.py preparar_base_datos
```

Versiones instaladas y comprobadas en este equipo:

- Python 3.13.14.
- Django 6.1.3.
- opencv-contrib-python 4.13.0.92, con el módulo `cv2.face`.
- NumPy 2.4.6.

Las dependencias están fijadas en `requirements.txt`. Referencias oficiales: [Django en PyPI](https://pypi.org/project/Django/6.1.3/) y [OpenCV Contrib en PyPI](https://pypi.org/project/opencv-contrib-python/4.13.0.92/).

## Funcionamiento

El sistema necesita cuatro coincidencias consecutivas para registrar una asistencia. La base de datos evita repetir la asistencia del mismo estudiante en la sesión diaria. La interfaz muestra los contadores y los últimos diez registros del día. La zona horaria es `America/Guayaquil`.

La cámara debe estar permitida en el navegador y libre de otras aplicaciones. Usa la dirección local indicada arriba; el acceso a la cámara desde otro equipo normalmente requiere HTTPS. La configuración actual corresponde a ejecución local con el servidor de desarrollo de Django.

## Archivos principales

- `asistencia/templates/asistencia/inicio.html`: interfaz.
- `asistencia/views.py`: procesamiento de imágenes y registro de asistencia.
- `asistencia/reconocedor.py`: carga y consulta del modelo facial.
- `asistencia/management/commands/`: comandos de preparación y entrenamiento.
- `config/settings.py`: configuración y rutas de fotografías/modelo.
- `db.sqlite3`: base de datos de estudiantes y asistencias.
- `fotografias_estudiantes/`: imágenes organizadas por código.
- `reconocimiento_facial/modelo_asistencia.yml`: modelo facial.

## Verificación

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py test
```

Las pruebas usan una base de datos temporal. Comprueban importación sin duplicados, rechazo de CSV incompleto, protección de rutas, conservación del modelo cuando faltan fotos, respuesta de la página, procesamiento de una imagen sin rostros y registro de asistencia tras cuatro coincidencias simuladas. Las coincidencias simuladas verifican la lógica de registro; no miden la precisión del reconocimiento facial.
