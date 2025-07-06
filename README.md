# SG-LLIE Fork — Entrenamiento desde Cero y Evaluación con Modelo Preentrenado

Este repositorio es un **fork modificado** del trabajo original [`minyan8/imagine`](https://github.com/minyan8/imagine), desarrollado como solución al [NTIRE 2025 Low Light Image Enhancement Challenge](https://codalab.lisn.upsaclay.fr/competitions/21636).

La presente versión incluye:
- Código adaptado para permitir **entrenamiento desde cero**.
- Instrucciones para el uso de un **modelo preentrenado con 485,000 iteraciones**.
- Guía de uso para ejecutar el entrenamiento y evaluación.
- Enlaces de descarga para los conjuntos de datos utilizados. LOLv1 - [Google Drive](https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view?usp=sharing) y LOLv2 con LOLv2-synthetic  - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)


> Nota: Este repositorio es utilizado para fines experimentales en un paper de investigación. Todo el crédito del modelo original es para sus autores.

---

## Requisitos de Instalación

1. Crear y activar entorno:

```bash
conda create --name imagine python=3.10
conda activate imagine

cd .\Enhacement\Test\
python extract_prior.py  

cd .\basicsr\
python train.py -opt options/Ntire25_LowLight.yml

cd .\Enhacement\
python test.py
