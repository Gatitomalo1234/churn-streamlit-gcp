#!/usr/bin/env bash
# Despliegue (o actualización) de la app en una VM Ubuntu 24.04 de Google Cloud.
#
# Uso, desde la terminal SSH de la VM:
#   curl -fsSL https://raw.githubusercontent.com/Gatitomalo1234/churn-streamlit-gcp/main/scripts/deploy_gcp.sh | bash
# o, si ya clonó el repositorio:
#   bash scripts/deploy_gcp.sh
#
# Es idempotente: se puede ejecutar de nuevo después de un cambio en el repositorio.
set -euo pipefail

REPO_URL="https://github.com/Gatitomalo1234/churn-streamlit-gcp.git"
REPO_DIR="$HOME/churn-streamlit-gcp"
IMAGE="ml-streamlit-app"
CONTAINER="ml-streamlit"
PORT=8501

# La e2-micro gratuita tiene 1 GB de RAM: se agrega 2 GB de swap para que
# la instalación de dependencias durante docker build no se quede sin memoria.
if [ "$(free -m | awk '/^Mem:/{print $2}')" -lt 2000 ] && ! swapon --show | grep -q /swapfile; then
  echo "==> 0/5 Creando swap de 2 GB (VM con poca RAM)"
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> 1/5 Instalando Docker y Git (si faltan)"
if ! command -v docker >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io git
fi
sudo systemctl enable --now docker

echo "==> 2/5 Descargando el código"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> 3/5 Construyendo la imagen $IMAGE"
sudo docker build -t "$IMAGE" .

echo "==> 4/5 Reemplazando el contenedor $CONTAINER"
sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
sudo docker run -d --name "$CONTAINER" --restart unless-stopped -p "$PORT:$PORT" "$IMAGE"

echo "==> 5/5 Verificando que la app responde"
for _ in $(seq 1 20); do
  if curl -fsS "http://localhost:$PORT/_stcore/health" >/dev/null 2>&1; then
    EXTERNAL_IP=$(curl -fsS -H "Metadata-Flavor: Google" \
      "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip" \
      2>/dev/null || echo "IP_EXTERNA_DE_LA_VM")
    sudo docker ps --filter "name=$CONTAINER"
    echo
    echo "App en línea: http://$EXTERNAL_IP:$PORT"
    echo "Si no abre desde su navegador, revise la regla de firewall para tcp:$PORT (README, sección 12)."
    exit 0
  fi
  sleep 3
done

echo "La app no respondió. Revise los logs con: sudo docker logs $CONTAINER" >&2
exit 1
