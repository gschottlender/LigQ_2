# Plan de despliegue de LigQ 2 con Docker

## Objetivo

Empaquetar la herramienta de linea de comandos, el backend FastAPI y el frontend
de LigQ 2 para que puedan ejecutarse de forma reproducible con Docker, sin que el
usuario tenga que instalar Conda, Python, Node.js, RDKit, BLAST, HMMER o PyTorch.

La solucion debe conservar los datos, representaciones, resultados e historial
aunque los contenedores se detengan o se actualicen.

## Arquitectura recomendada

La opcion mas practica es usar Docker Compose con dos servicios activos y una
imagen Python compartida por la API y la CLI:

```text
Navegador
   |
   v
Frontend Nginx ---- /api ----> FastAPI + pipeline Python
                                  |
CLI Docker -----------------------+
                                  |
                        Datos y resultados persistentes
```

Servicios:

- `api`: FastAPI y todo el pipeline de LigQ 2.
- `web`: frontend compilado y servido por Nginx.
- `cli`: servicio ejecutable bajo demanda, construido con la misma imagen que
  `api`.

La API y la CLI deben usar la misma imagen para garantizar que ambas ejecuten
las mismas versiones de Python, RDKit, BLAST, HMMER, PyTorch y el resto de las
dependencias.

## Imagen de Python

La imagen del pipeline puede basarse en Micromamba o Miniforge y debe:

- Instalar el entorno definido por `environment.yml`.
- Copiar el codigo de LigQ 2 en `/app`.
- Incluir BLAST, HMMER, RDKit, PyTorch y las dependencias de FastAPI.
- Ejecutar Uvicorn desde `/app/gui/backend`.
- Escuchar en `0.0.0.0:8000`, no solamente en `127.0.0.1`.
- Ejecutarse con un usuario sin privilegios.
- Mantener el codigo dentro de la imagen y los datos fuera de ella.

El backend ya facilita este esquema porque ejecuta los scripts mediante
`sys.executable`. Los trabajos lanzados por FastAPI utilizaran el mismo entorno
Python que el servidor.

Para la primera version conviene publicar una imagen orientada a CPU. El soporte
para GPU NVIDIA puede agregarse despues mediante una imagen y un archivo Compose
opcionales, junto con NVIDIA Container Toolkit.

## Imagen del frontend

El frontend debe construirse con un Dockerfile multietapa:

1. Una etapa con Node.js 24 ejecuta `npm ci` y `npm run build`.
2. Una imagen pequena de Nginx recibe solamente el contenido de `dist/`.
3. Nginx sirve los archivos estaticos y redirige `/api` hacia `api:8000`.
4. Las rutas que no correspondan a archivos deben volver a `index.html` para
   permitir la navegacion de React.
5. Debe configurarse un limite de carga suficiente para bases de compuestos
   grandes mediante `client_max_body_size`.
6. Los assets versionados pueden usar cache prolongada, incluyendo el archivo
   WASM de RDKit.

Vite debe utilizarse para desarrollo, pero no como servidor de produccion.

## Datos persistentes

Las bases no deben incorporarse a la imagen. La carpeta `databases/` local
actualmente ocupa aproximadamente 11 GB y puede cambiar cuando se agregan bases,
representaciones o caches.

Rutas que deben montarse como volumen:

```text
/app/databases                  Bases, representaciones y caches
/app/results                    Resultados e historial
/app/gui/backend/uploads        Archivos cargados desde la GUI
/cache/huggingface              Modelos y descargas de Hugging Face
/work                           FASTA y resultados utilizados desde la CLI
```

Para una instalacion local conviene usar carpetas montadas, por ejemplo:

```text
docker-data/databases/
docker-data/results/
docker-data/uploads/
docker-data/huggingface/
work/
```

Esto simplifica backups, inspeccion, recuperacion y transferencia a otra
maquina. Los permisos deben mapearse al UID y GID del usuario para evitar que
Docker genere archivos pertenecientes a `root`.

## Experiencia de uso esperada

Inicio de la aplicacion:

```bash
docker compose up -d
```

La GUI quedaria disponible en:

```text
http://localhost:8080
```

Ejecucion de la CLI con el mismo runtime que usa la GUI:

```bash
docker compose run --rm cli \
  --input-fasta /work/queries.fasta \
  --output-dir /work/results \
  --ligand-provider zinc
```

Tambien conviene agregar comandos auxiliares sencillos:

```text
make start       Iniciar la aplicacion
make stop        Detenerla
make logs        Mostrar logs
make init-data   Preparar o descargar las bases iniciales
make cli         Ejecutar la herramienta de linea de comandos
```

El primer inicio debe informar claramente la descarga de bases y modelos. Esos
archivos deben quedar en los volumenes para que no se descarguen nuevamente al
recrear los contenedores.

## Concurrencia y trabajos

Antes de considerar el despliegue completamente robusto deben resolverse tres
aspectos del comportamiento actual.

### Directorios temporales unicos

Las busquedas pueden compartir actualmente `temp_results`. Cada trabajo deberia
usar una ruta propia, por ejemplo:

```text
/app/temp_results/<job_id>/
```

Esto evita que una busqueda elimine o sobrescriba los archivos intermedios de
otra.

### Control de concurrencia

La creacion de bases y representaciones modifica el volumen compartido. Estas
operaciones deben ejecutarse en una cola o bajo un bloqueo de escritura. Tampoco
debe iniciarse una busqueda contra una representacion que se esta regenerando.

Para una instalacion local y de un solo usuario es suficiente comenzar con una
cola interna y permitir un solo trabajo pesado simultaneo. Redis, RQ o Celery
pueden evaluarse mas adelante si la aplicacion pasa a ser multiusuario.

### Persistencia del estado

FastAPI conserva actualmente el estado de los trabajos en memoria. Conviene
persistir el registro de jobs en SQLite, dentro de un volumen. Si el contenedor
se reinicia, los trabajos que estaban activos pueden marcarse como
`interrupted` o `failed`, mientras que los resultados terminados siguen
disponibles desde el disco.

Tambien debe configurarse un tiempo de apagado suficiente y manejar las senales
de terminacion para cerrar correctamente los procesos hijos.

## Red y seguridad

- Solamente el servicio `web` necesita publicar un puerto al host.
- `api:8000` puede permanecer accesible solo dentro de la red de Compose.
- Nginx debe ser el unico punto de entrada y enviar `/api` hacia FastAPI.
- Los tokens opcionales, como `HF_TOKEN`, deben cargarse desde `.env` o secretos
  de Docker y nunca copiarse dentro de la imagen.
- Los nombres y versiones de las imagenes deben fijarse con tags reproducibles.
- Se deben agregar healthchecks para Nginx y `/api/health`.

## Distribucion

Lo ideal es construir y publicar las imagenes versionadas en GitHub Container
Registry mediante GitHub Actions. El usuario final solamente necesitaria Docker
Engine o Docker Desktop y podria ejecutar:

```bash
docker compose pull
docker compose up -d
```

Esto evita que cada usuario tenga que resolver localmente el entorno Conda, que
es una de las partes mas pesadas y sensibles de la instalacion.

Para mejorar la reproducibilidad, el entorno Python deberia generar un lock para
`linux-64`. El frontend ya puede reproducirse mediante `package-lock.json` y
`npm ci`.

## Estrategia de implementacion

Orden recomendado:

1. Hacer que cada job tenga directorios temporales unicos.
2. Agregar bloqueo o cola para operaciones que modifican bases y
   representaciones.
3. Persistir el estado minimo de trabajos en SQLite.
4. Crear la imagen comun de Python para API y CLI.
5. Crear la imagen multietapa del frontend con Nginx.
6. Definir Compose, volumenes, healthchecks y variables de entorno.
7. Agregar comandos simples de inicio, parada, logs, inicializacion y CLI.
8. Probar persistencia despues de reiniciar y recrear contenedores.
9. Publicar imagenes CPU versionadas en GitHub Container Registry.
10. Agregar soporte GPU opcional si resulta necesario.

## Validacion minima

La solucion debe probar al menos estos casos:

- Inicio limpio sin bases descargadas.
- Descarga y reutilizacion de datos despues de reiniciar.
- Ejecucion CLI con un FASTA pequeno.
- Busqueda completa desde la GUI.
- Busqueda en modo `known-only`.
- Carga de una base personalizada.
- Creacion exitosa y fallida de una representacion.
- Persistencia de resultados e historial al recrear contenedores.
- Mensajes de error visibles cuando falla un proceso hijo.
- Uso simultaneo controlado sin corrupcion de archivos temporales.

## Alternativas que conviene evitar

- Ejecutar Vite como servidor de produccion.
- Instalar dependencias cada vez que se inicia el contenedor.
- Incorporar los 11 GB de bases dentro de la imagen.
- Mantener imagenes Python diferentes para CLI y FastAPI.
- Ejecutar Nginx, Vite y FastAPI como procesos manuales dentro de un unico
  contenedor.
- Permitir varios trabajos sobre el mismo `temp_results`.
- Depender exclusivamente del estado en memoria para mostrar el historial.

Esta arquitectura mantiene el uso cotidiano simple, conserva la compatibilidad
entre CLI y GUI y permite agregar escalabilidad mas adelante sin introducir
desde el comienzo componentes innecesarios.
