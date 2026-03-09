# Alvaro_Fernandez-e-Inigo_Del_Mazo---Proyecto_generacion_de_musica---PIA
# Generador de Música Soul con IA

**Integrantes:** Álvaro Fernández e Iñigo Del Mazo

## Descripción
Esta aplicación web genera pistas de música instrumental del género Soul basándose en descripciones textuales. El usuario puede seleccionar un estilo predefinido y añadir instrumentos extra a su gusto. El sistema utiliza un modelo de Inteligencia Artificial de código abierto ejecutado 100% en local.

## Modelo Base
Hemos elegido el modelo `facebook/musicgen-small`. Es un modelo Transformer autorregresivo con 300 millones de parámetros. Lo seleccionamos porque es una versión reducida y potente que permite generar muestras de audio de alta calidad sin requerir recursos computacionales inasumibles para una ejecución local.

## Técnica de Adaptación
Hemos optado por la Opción A (Fine-Tuning Tradicional). Para hacerlo de forma eficiente y evitar el colapso del modelo, utilizamos la técnica LoRA mediante la librería PEFT. Esta técnica nos permitió entrenar una pequeña capa externa adaptada a nuestro estilo Soul sin destruir los pesos originales del modelo base. El entrenamiento se realizó en Google Colab para solventar limitaciones de hardware, y la ejecución final (inferencia) se realiza en local a través de una interfaz web.

## Dataset
Creamos un dataset propio recopilando 25 clips de música Soul instrumental (sin voz) cortados a aproximadamente 30 segundos cada uno. Para el etiquetado, generamos un archivo `metadata.jsonl` que relaciona cada audio `.wav` con una descripción textual detallada de los instrumentos y el ritmo (por ejemplo: "Soul romántico, ritmo suave, bajo profundo y cuerdas").

## Instrucciones de Instalación y Ejecución

Para probar nuestra aplicación, sigue estos pasos en tu terminal de comandos:

```bash
git clone [https://github.com/tu-usuario/Alvaro-Fernandez-e-I-igo-Del-Mazo---Proyecto-generaci-n-de-m-sica---PIA.git](https://github.com/tu-usuario/Alvaro-Fernandez-e-I-igo-Del-Mazo---Proyecto-generaci-n-de-m-sica---PIA.git)
cd Alvaro-Fernandez-e-I-igo-Del-Mazo---Proyecto-generaci-n-de-m-sica---PIA
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
python src/app.py
