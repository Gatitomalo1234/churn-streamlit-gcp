# Evidencias del despliegue

Guarde aquí las capturas que demuestran cada criterio de la rúbrica y márquelas en la lista.

| # | Evidencia | Archivo sugerido | Criterio |
|---|---|---|---|
| ☐ 1 | `train.py` ejecutado: comparación de modelos y métricas de test | `01_train.png` | Modelamiento |
| ☑ 2 | `docker build` terminado sin errores en la VM | `02_docker_build_y_ps.png` | Docker |
| ☑ 3 | `docker ps` con el contenedor `ml-streamlit` en estado *Up* | `02_docker_build_y_ps.png` | Docker |
| ☐ 4 | Instancia de Compute Engine con IP externa (consola de GCP) | `04_vm_gcp.png` | GCP |
| ☐ 5 | Regla de firewall `allow-streamlit-8501` (tcp:8501) | `05_firewall.png` | GCP |
| ☑ 6 | App abierta en `http://IP_EXTERNA:8501`, con la IP visible en la barra del navegador | `06_app_ip_externa.png` | GCP · Streamlit |
| ☐ 7 | Caso de alto riesgo (78,5% · Churn) en el simulador | `07_caso_alto_riesgo.png` | Streamlit |
| ☐ 8 | Carga masiva de `data/pruebas/muestra_500_con_churn.xlsx` | `08_carga_masiva.png` | Streamlit |
| ☐ 9 | VM detenida o eliminada al terminar | `09_vm_detenida.png` | Cierre |

Los casos esperados de las pruebas están en la sección 13 del README principal.

## Registro del despliegue

- **Fecha:** 9 de noviembre de 2026.
- **VM:** `churn-streamlit-vm` (e2-micro, `us-central1-c`, Ubuntu 24.04, disco estándar de 20 GB), dentro del nivel gratuito de Google Cloud.
- **Despliegue:** `scripts/deploy_gcp.sh` terminó con *Successfully built* y *Successfully tagged ml-streamlit-app:latest*; el contenedor `ml-streamlit` quedó en estado *Up* con el puerto `0.0.0.0:8501->8501/tcp`.
- **Acceso externo:** la app abrió en `http://35.208.96.179:8501`. La IP es efímera y cambia cada vez que se enciende la VM.
