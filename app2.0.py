import gradio as gr
from transformers import pipeline

pipe = pipeline("text-to-audio", model="./mi_modelo_soul")

def crear_soul(descripcion, duracion, instrumento):
    prompt_final = descripcion
    if instrumento != "Ninguno":
        prompt_final = f"{descripcion}, destaca {instrumento}"
        
    tokens = int(duracion * 50)
    
    resultado = pipe(prompt_final, forward_params={"max_new_tokens": tokens})
    audio_data = resultado["audio"].squeeze()
    frecuencia = resultado["sampling_rate"]
    
    return (frecuencia, audio_data)

interfaz = gr.Interface(
    fn=crear_soul,
    inputs=[
        gr.Textbox(lines=3, placeholder="Ejemplo: ritmo suave, bajo profundo..."),
        gr.Slider(minimum=1, maximum=30, step=1, value=5, label="Duración (segundos)"),
        gr.Radio(["Ninguno", "Saxofón", "Piano", "Bajo"], label="Instrumento destacado", value="Ninguno")
    ],
    outputs=gr.Audio(label="Vuestro Soul Generado"),
    title="Generador de Soul IA",
    description="Modelo especializado entrenado por Iñigo y Álvaro."
)

interfaz.launch()