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
                "Soul clásico, voz potente, metales y ritmo enérgico",
                "Funk soul, ritmo muy marcado, bajo y batería bailable",
                "Soul romántico, ritmo suave, bajo profundo y cuerdas",
                "R&B clásico, piano blues, tempo medio",
                "Neo-soul oscuro, ritmo retro, trompetas y bajo destacado",
                "Soul alegre, teclado funk, ritmo movido",
                "Soul tradicional, melodía vocal suave, acompañamiento acústico",
                "R&B de los 60, percusión suave, cuerdas de fondo",
                "R&B pop, balada emotiva, piano y sintetizadores suaves",
                "R&B moderno, piano principal, ritmo urbano lento"
            ], 
            value="Soul romántico, ritmo suave, bajo profundo y cuerdas",
            label="Estilo principal de la pista"
        ),
        gr.Slider(minimum=1, maximum=30, step=1, value=5, label="Duración (segundos)"),
        gr.CheckboxGroup(
            choices=["saxofón", "piano", "bajo eléctrico", "batería acústica", "guitarra eléctrica", "trompeta", "sintetizador"], 
            label="Instrumentos extra (puedes marcar varios)"
        )
    ],
    outputs=gr.Audio(label="Vuestro Soul Generado")
)

interfaz.launch()