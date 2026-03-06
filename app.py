import gradio as gr
from transformers import pipeline

# Cargamos vuestro modelo local
pipe = pipeline("text-to-audio", model="./mi_modelo_soul")

def crear_soul(descripcion):
    print("Generando pista...")
    
    # max_new_tokens=256 genera unos 5 segundos. Puedes subirlo.
    resultado = pipe(descripcion, forward_params={"max_new_tokens": 256})
    audio_data = resultado["audio"].squeeze()
    frecuencia = resultado["sampling_rate"]
    
    # Gradio reproduce el audio directamente si le pasamos esta estructura
    return (frecuencia, audio_data)

# Creamos la ventana visual
interfaz = gr.Interface(
    fn=crear_soul,
    inputs=gr.Textbox(lines=3, placeholder="Ejemplo: ritmo suave, bajo profundo, saxofón brillante..."),
    outputs=gr.Audio(label="Vuestro Soul Generado"),
    title="Generador de Soul IA",
    description="Modelo especializado entrenado por Iñigo y Álvaro."
)

# Arrancamos el servidor local
interfaz.launch()