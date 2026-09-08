# LinkedIn — post en español (principal)

> Formato: post de texto. Sin imagen obligatoria; si quieres, adjunta
> `results/cpu_all.png` (el Pareto calidad vs velocidad) — sube la imagen, no
> pongas el link, LinkedIn penaliza links externos en el cuerpo. Pon el link al
> repo en el **primer comentario**.

---

Una empresa quiere un LLM en su propio servidor. Los datos no pueden salir del
edificio, o la factura por token en la nube no cierra. Tienen una máquina normal:
8 núcleos de CPU, 30 GB de RAM, **sin GPU**.

¿Qué modelo corre ahí? ¿A qué cuantización? ¿Alcanza para producción? ¿Y vale la
pena comprar una GPU?

No encontré esos números medidos en un solo lugar, así que los medí. 7 modelos
instruct (2–8B), 2–3 niveles de cuantización GGUF cada uno, en esa máquina exacta.
Velocidad de generación, RAM, tiempo de carga, concurrencia, ruido entre corridas,
dos motores (llama.cpp vs Ollama) y la comparación contra una GPU (Tesla P100).

Los tres hallazgos que me cambiaron la intuición:

**1. En CPU, Q4_K_M es la cuantización correcta — punto.** Todos los modelos
puntuaron igual o mejor en calidad en Q4_K_M que en Q8_0, corriendo 20–40% más
rápido y con menos RAM. Q8 no compra nada en CPU. (En GPU sí sirve.)

**2. Hay un salto de calidad marcado en los 7B, y un modelo se lo queda.** Toda
la clase de 3B se agrupa en ~0.70 en mi prueba. qwen2.5-7b salta a 0.87 — y lo
mantiene en Q3_K_M, que entra en 5.5 GB. llama-3.1-8b y mistral-7b NO dan ese
salto: pagas 2× el tamaño sin ganar calidad.

**3. Una caja de CPU es una unidad de servicio de un solo usuario.** El
throughput agregado es plano sin importar la concurrencia — una sola petición ya
satura todos los núcleos. Para N usuarios concurrentes en vivo: ~N cajas, o una
GPU. La GPU pierde en costo por token (≈8× la renta, ≈4× la velocidad) pero gana
en latencia, en prefill para RAG (20–30×) y en modelos de 14B que la CPU no
corre útilmente.

Todo el código, la tabla completa y una guía de decisión (tu caja + tu caso de
uso → el modelo, la cuantización y el motor) están en el repo. Link en el primer
comentario.

Lo escribí también en inglés. Si estás montando inferencia on-premise y algún
número no te cuadra con lo que ves en tu hardware, me interesa saberlo.

#LLM #MLOps #InferenciaOnPremise #EdgeAI #Ollama #llamacpp

---

## Primer comentario (pegar aparte)

Repo (código + resultados + guía de decisión):
https://github.com/chrono-glitch/onprem-llm-bench

Writeup completo:
- ES: https://github.com/chrono-glitch/onprem-llm-bench/blob/main/WRITEUP.es.md
- EN: https://github.com/chrono-glitch/onprem-llm-bench/blob/main/WRITEUP.en.md
