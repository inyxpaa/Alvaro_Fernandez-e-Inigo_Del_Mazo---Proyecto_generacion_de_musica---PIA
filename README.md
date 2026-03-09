# Alvaro_Fernandez-e-Inigo_Del_Mazo---Proyecto_generacion_de_musica---PIA
# Generador de Música Soul con IA

**Integrantes:** Álvaro Fernández e Iñigo Del Mazo

## Descripción
Esta aplicación web genera pistas de música instrumental Soul. El usuario escribe o elige un estilo. También puede añadir instrumentos extra. El sistema usa un modelo de Inteligencia Artificial. Todo funciona 100% en local.

## Modelo Base
Usamos el modelo `facebook/musicgen-small`. Es un modelo Transformer autorregresivo. Tiene 300 millones de parámetros. Es un modelo ligero. Genera audio de alta calidad sin colapsar ordenadores normales.

## Técnica de Adaptación
Elegimos la Opción A (Fine-Tuning Tradicional). Usamos la técnica LoRA con la librería PEFT. Esta técnica entrena una capa externa pequeña. Protege los pesos originales del modelo base. Entrenamos el modelo en Google Colab para evitar problemas de memoria. La ejecución final se hace en local con una web.

## Dataset
Creamos un dataset propio. Recopilamos 25 clips de música Soul instrumental sin voz. Cada clip dura unos 30 segundos. Creamos el archivo `metadata.jsonl`. Este archivo enlaza cada audio con un texto exacto. Ejemplo: "Soul romántico, ritmo suave, bajo profundo y cuerdas".

## Requisitos Previos
* Python 3.10 o superior.

## Instrucciones de Instalación y Ejecución
Sigue estos pasos exactos en tu terminal:

1. **Clonar el repositorio:**
```bash
git clone [https://github.com/tu-usuario/Alvaro_Fernandez-e-Inigo_Del_Mazo---Proyecto_generacion_de_musica---PIA.git](https://github.com/tu-usuario/Alvaro_Fernandez-e-Inigo_Del_Mazo---Proyecto_generacion_de_musica---PIA.git)
cd Alvaro_Fernandez-e-Inigo_Del_Mazo---Proyecto_generacion_de_musica---PIA
