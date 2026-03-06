import gradio as gr
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from peft import PeftModel

procesador = AutoProcessor.from_pretrained("facebook/musicgen-small")
modelo_base = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
modelo = PeftModel.from_pretrained(modelo_base, "./musicgen-soul-lora-alvaro-inigo")

def crear_soul(estilo, duracion, instrumentos):
    prompt_final = estilo
    
    if len(instrumentos) > 0:
        lista_instrumentos = ", ".join(instrumentos)
        prompt_final = f"{estilo}, {lista_instrumentos}"
        
    entradas = procesador(
        text=[prompt_final],
        padding=True,
        return_tensors="pt"
    )
    
    tokens = int(duracion * 50)
    
    salida_audio = modelo.generate(
        **entradas,
        max_new_tokens=tokens,
        guidance_scale=2.5,
        temperature=0.8,
        do_sample=True
    )
    
    audio_data = salida_audio[0, 0].cpu().numpy()
    frecuencia = modelo.config.audio_encoder.sampling_rate
    
    return (frecuencia, audio_data)

interfaz = gr.Interface(
    fn=crear_soul,
    inputs=[
        gr.Radio(
            choices=[
                "Soul clásico, ritmo animado", 
                "R&B moderno, piano melancólico, tempo lento", 
                "Soul suave, bajo profundo, ritmo relajado", 
                "R&B enérgico, metales brillantes, estilo retro"
            ], 
            value="Soul suave, bajo profundo, ritmo relajado",
            label="Estilo principal de la pista"
        ),
        gr.Slider(minimum=1, maximum=30, step=1, value=5, label="Duración (segundos)"),
        gr.CheckboxGroup(
            choices=["saxofón", "piano", "bajo eléctrico", "batería acústica", "guitarra eléctrica"], 
            label="Instrumentos extra (puedes marcar varios)"
        )
    ],
    outputs=gr.Audio(label="Vuestro Soul Generado")
)

interfaz.launch()