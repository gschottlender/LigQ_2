# Instalación de LigQ 2 Web en Nodo0

Esta guía despliega la versión web pública y restringida de LigQ 2 usando las
imágenes publicadas para un commit específico. Solamente el frontend queda
conectado a Traefik; la API permanece en una red interna y las bases se montan
en modo de solo lectura durante la ejecución normal.

## 1. Estructura recomendada

```text
/home/dockeradmin/ligq2/
├── repo/                    # repositorio clonado
├── docker-compose.yml       # configuración exclusiva de Nodo0
├── nginx.nodo0.conf
└── .env

/data/ligq2/
├── databases/
├── results/
├── uploads/
├── state/
├── temp/
├── huggingface/
└── secrets/
```

Acceso:

```bash
ssh cranex
sudo -iu glyco
ssh nodo0
sudo -iu dockeradmin
```

Verificar la red real de Traefik:

```bash
docker inspect traefik_nodo0 --format '{{json .NetworkSettings.Networks}}'
docker network inspect internal-nodo0-web
```

Antes de continuar, confirmar que el dominio elegido y el router
`ligq2-nodo0` no estén usados por otra aplicación.

## 2. Preparar directorios y secreto

```bash
sudo install -d -o 1000 -g 1000 -m 0750 \
  /data/ligq2/databases \
  /data/ligq2/results \
  /data/ligq2/uploads \
  /data/ligq2/state \
  /data/ligq2/temp \
  /data/ligq2/huggingface

sudo install -d -o dockeradmin -g dockeradmin -m 0700 /data/ligq2/secrets

umask 077
openssl rand -hex 32 > /data/ligq2/secrets/session_secret
```

Reservar al menos unos 20 GB libres durante la descarga inicial.

## 3. Configurar imágenes y dominio

Obtener el commit actual del repositorio:

```bash
cd /home/dockeradmin/ligq2/repo
git checkout main
git pull --ff-only origin main
git rev-parse --short=7 HEAD
```

Cada push a `main` publica las imágenes
`ghcr.io/gschottlender/ligq-2-api:sha-XXXXXXX` y
`ghcr.io/gschottlender/ligq-2-web:sha-XXXXXXX`. Esperar a que GitHub Actions
termine y usar el tag del commit obtenido, no el tag mutable `main`.

Por ejemplo, para el commit `ce84f24`, crear
`/home/dockeradmin/ligq2/.env`:

```dotenv
LIGQ_API_IMAGE=ghcr.io/gschottlender/ligq-2-api:sha-ce84f24
LIGQ_WEB_IMAGE=ghcr.io/gschottlender/ligq-2-web:sha-ce84f24

# Confirmar que sea un dominio único antes de usarlo.
LIGQ_HOST=ligq.infra.cluster.qb.fcen.uba.ar

# Puerto local exclusivamente para diagnóstico.
LIGQ_WEB_BIND=127.0.0.1
LIGQ_WEB_PORT=18081
```

Usar siempre el mismo tag para API y frontend.

```bash
chmod 600 /home/dockeradmin/ligq2/.env
```

## 4. Adaptar Nginx para Traefik

```bash
cd /home/dockeradmin/ligq2
cp repo/docker/nginx.web.conf nginx.nodo0.conf
```

Dentro de `location /api/` en `nginx.nodo0.conf`, reemplazar las tres cabeceras
proxy por:

```nginx
proxy_set_header X-Real-IP $http_x_real_ip;
proxy_set_header X-Forwarded-For $http_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $http_x_forwarded_proto;
```

Esto conserva la IP real enviada por Traefik, necesaria para que el límite de
solicitudes funcione por usuario.

## 5. Crear el Compose de Nodo0

Guardar lo siguiente en `/home/dockeradmin/ligq2/docker-compose.yml`:

```yaml
services:
  api:
    container_name: ligq2_api_nodo0
    environment:
      LIGQ_SESSION_SECRET_FILE: /run/secrets/ligq_session_secret
      LIGQ_SESSION_COOKIE_SECURE: "true"
      LIGQ_TRUST_PROXY_HEADERS: "true"
      ALLOWED_ORIGINS: "https://${LIGQ_HOST}"
    secrets:
      - ligq_session_secret
    volumes:
      - /data/ligq2/databases:/app/databases:ro
      - /data/ligq2/results:/app/results
      - /data/ligq2/uploads:/app/gui/backend/uploads
      - /data/ligq2/state:/app/state
      - /data/ligq2/temp:/app/temp_results
    networks:
      - ligq_internal
    cpus: "0.85"
    mem_limit: 1700m
    pids_limit: 256
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    logging:
      options:
        max-size: "10m"
        max-file: "3"

  web:
    container_name: ligq2_web_nodo0
    volumes:
      - /home/dockeradmin/ligq2/nginx.nodo0.conf:/etc/nginx/conf.d/default.conf:ro
    networks:
      - ligq_internal
      - web
    expose:
      - "8080"
    read_only: true
    tmpfs:
      - /tmp
      - /var/cache/nginx
    cpus: "0.15"
    mem_limit: 256m
    pids_limit: 64
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    logging:
      options:
        max-size: "10m"
        max-file: "3"
    labels:
      - traefik.enable=true
      - traefik.docker.network=internal-nodo0-web

      - traefik.http.routers.ligq2-nodo0.entrypoints=websecure
      - "traefik.http.routers.ligq2-nodo0.rule=Host(`${LIGQ_HOST}`)"
      - traefik.http.routers.ligq2-nodo0.tls=true
      - traefik.http.routers.ligq2-nodo0.tls.certresolver=leresolver
      - traefik.http.routers.ligq2-nodo0.service=ligq2-nodo0
      - traefik.http.routers.ligq2-nodo0.middlewares=ligq2-secure-nodo0
      - traefik.http.services.ligq2-nodo0.loadbalancer.server.port=8080

      - traefik.http.middlewares.ligq2-secure-nodo0.headers.framedeny=true
      - traefik.http.middlewares.ligq2-secure-nodo0.headers.contenttypenosniff=true
      - traefik.http.middlewares.ligq2-secure-nodo0.headers.referrerpolicy=no-referrer
      - traefik.http.middlewares.ligq2-secure-nodo0.headers.stsseconds=31536000

  prepare-data:
    volumes:
      - /data/ligq2/databases:/app/databases
      - /data/ligq2/huggingface:/cache/huggingface
    networks:
      - ligq_admin
    cpus: "1.0"
    mem_limit: 2g
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

  validate-data:
    volumes:
      - /data/ligq2/databases:/app/databases:ro
    network_mode: none
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

networks:
  ligq_internal:
    name: ligq2_nodo0_internal
    internal: true

  ligq_admin:
    name: ligq2_nodo0_admin

  web:
    external: true
    name: internal-nodo0-web

secrets:
  ligq_session_secret:
    file: /data/ligq2/secrets/session_secret
```

La API conserva las restricciones del modo web: una búsqueda activa, sin
administración de recursos, bases y cachés de predicciones de solo lectura,
límite de FASTA, tiempo máximo, retención corta y rate limiting. Además, el
backend pasa explícitamente `--n-workers 1` al pipeline para que BLAST y HMMER
usen un solo worker. Los límites de CPU y RAM del Compose pueden ajustarse
únicamente después de medir el uso real y coordinarlo con la administración del
Nodo0.

## 6. Descargar datos y levantar

Los siguientes comandos combinan el stack web del repositorio con la
configuración específica de Nodo0:

```bash
cd /home/dockeradmin/ligq2
export COMPOSE_FILE=repo/compose.web.yml:docker-compose.yml

docker compose --env-file .env config
docker compose --env-file .env pull
docker compose --env-file .env --profile admin run --rm prepare-data
docker compose --env-file .env --profile admin run --rm validate-data
```

La validación debe mostrar `ready`. El paso `prepare-data` descarga las bases y
los cachés ECFP/FCFP obligatorios para la versión web; evita iniciar con el error
de archivos faltantes.

Probar primero en primer plano:

```bash
docker compose --env-file .env up api web
```

Al mismo tiempo, observar Traefik desde otra terminal:

```bash
cd /home/dockeradmin/traefik
docker compose logs --tail=20 -f
```

Si funciona, detener el primer plano con `Ctrl+C` y levantar definitivamente:

```bash
cd /home/dockeradmin/ligq2
export COMPOSE_FILE=repo/compose.web.yml:docker-compose.yml
docker compose --env-file .env up -d api web
docker compose --env-file .env ps
```

Comprobar primero por el puerto local y luego por HTTPS:

```bash
curl -fsS http://127.0.0.1:18081/health
curl -fsS https://ligq.infra.cluster.qb.fcen.uba.ar/api/health
```

Reemplazar el dominio del segundo comando si se eligió otro.

## 7. Actualizar después de cambios en el repositorio

Primero hacer commit y push de los cambios a `main`. El workflow de GitHub
publicará las dos imágenes. En Nodo0:

```bash
cd /home/dockeradmin/ligq2/repo
git status --short
git pull --ff-only origin main
git rev-parse --short=7 HEAD
```

Si `git status` muestra cambios locales, no actualizar hasta resolverlos.
Esperar a que GitHub Actions publique ambos tags `sha-XXXXXXX`.

Guardar la configuración anterior:

```bash
cd /home/dockeradmin/ligq2
cp .env .env.rollback
```

Editar `.env` y cambiar **ambas** imágenes al nuevo tag. Luego:

```bash
export COMPOSE_FILE=repo/compose.web.yml:docker-compose.yml

docker compose --env-file .env pull
docker compose --env-file .env --profile admin run --rm validate-data
docker compose --env-file .env up -d api web
docker compose --env-file .env ps
docker compose --env-file .env logs --tail=100 api web
```

Para cambios normales de código no hay que volver a ejecutar `prepare-data`.
Si la validación de datos falla con una versión nueva, no reiniciar la
aplicación hasta revisar los requisitos de datos de esa versión.

Comprobar nuevamente:

```bash
curl -fsS http://127.0.0.1:18081/health
curl -fsS https://ligq.infra.cluster.qb.fcen.uba.ar/api/health
```

### Rollback

Si la versión nueva falla:

```bash
cd /home/dockeradmin/ligq2
export COMPOSE_FILE=repo/compose.web.yml:docker-compose.yml
cp .env.rollback .env
docker compose --env-file .env up -d api web
docker compose --env-file .env ps
```

## Reglas de seguridad

- No conectar la API directamente a la red de Traefik.
- No publicar el puerto `8000` de la API.
- Mantener `/data/ligq2/databases` en modo de solo lectura durante la ejecución.
- No guardar secretos dentro del repositorio.
- No usar el tag mutable `main` en producción; usar `sha-XXXXXXX`.
- No ejecutar `docker compose down -v`.
- No borrar `/data/ligq2` durante una actualización.
- No ejecutar limpiezas globales como `docker system prune -a` en Nodo0.
- Revisar logs de la aplicación y Traefik antes de levantar con `-d`.
