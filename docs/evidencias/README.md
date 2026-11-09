# Evidencias del despliegue

Guarde aquí las capturas que demuestran cada criterio de la rúbrica y márquelas en la lista.

| # | Evidencia | Archivo sugerido | Criterio |
|---|---|---|---|
| ☐ 1 | `train.py` ejecutado: comparación de modelos y métricas de test | `01_train.png` | Modelamiento |
| ☐ 2 | `docker build` terminado sin errores en la VM | `02_docker_build.png` | Docker |
| ☐ 3 | `docker ps` con el contenedor `ml-streamlit` en estado *Up* | `03_docker_ps.png` | Docker |
| ☐ 4 | Instancia de Compute Engine con IP externa (consola de GCP) | `04_vm_gcp.png` | GCP |
| ☐ 5 | Regla de firewall `allow-streamlit-8501` (tcp:8501) | `05_firewall.png` | GCP |
| ☐ 6 | App abierta en `http://IP_EXTERNA:8501`, con la IP visible en la barra del navegador | `06_app_ip_externa.png` | GCP · Streamlit |
| ☐ 7 | Caso de alto riesgo (78,5% · Churn) en el simulador | `07_caso_alto_riesgo.png` | Streamlit |
| ☐ 8 | Carga masiva de `data/pruebas/muestra_500_con_churn.xlsx` | `08_carga_masiva.png` | Streamlit |
| ☐ 9 | Prueba cruzada: la otra pareja con la app corriendo desde su propia ejecución | `09_prueba_cruzada.png` | README y prueba cruzada |
| ☐ 10 | VM detenida o eliminada al terminar | `10_vm_detenida.png` | Cierre |

Los casos esperados de las pruebas están en la sección 13 del README principal.
