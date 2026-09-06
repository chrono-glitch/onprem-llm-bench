# Servir LLMs on-premise sin GPU: los números y cómo elegir

*Borrador — track de CPU. La comparación con GPU (Kaggle T4/P100) es una sección aparte.*

## El problema real

Una empresa quiere un LLM corriendo en su propio hardware — los datos no pueden
salir del edificio, o una cuenta en la nube por token no le cierra. Tienen un
servidor normal: 8 núcleos de CPU, ~30 GB de RAM, **sin GPU**. ¿Qué modelo
abierto, a qué nivel de cuantización, realmente funciona ahí — y alcanza para
producción?

Probé 7 modelos instruct (2–8 B) a 2–3 niveles de cuantización GGUF cada uno, en
exactamente esa máquina (8 vCPU AMD EPYC 7B12, `llama-cpp-python`), midiendo
velocidad de generación, RAM, tiempo de carga, y una prueba de calidad de 30
ítems (opción múltiple de razonamiento + seguir formato — un termómetro, no un
benchmark académico).

## Los tres hallazgos que importan

### 1. En CPU, `Q4_K_M` es la cuantización correcta — punto

Todos los modelos puntuaron **igual o mejor en calidad en Q4_K_M que en Q8_0**,
corriendo 20–40 % más rápido y usando menos RAM. Q8 no compra nada en CPU.
`Q3_K_M` es el piso real: bien para algunos modelos (qwen2.5-7b: 0.87 tanto en Q3
como en Q4), un golpe de ~4 puntos para otros (llama), no rompió nada.

### 2. Hay un salto de calidad marcado en los 7 B — y un modelo se lo queda

Toda la clase de 3 B se agrupó alrededor de **0.70**. `qwen2.5-7b` saltó a
**0.87** — y lo mantuvo en Q3_K_M, que entra en **5.5 GB** y corre a ~7 tok/s en
CPU. Si la máquina tiene la RAM y el caso de uso tolera ~7 tok/s (agentes, batch,
asíncrono), esta es la elección. `mistral-7b` y `llama-3.1-8b` **no** dieron ese
salto — llama-3.1-8b puntuó 0.67, ni mejor que el 3.2-3B con 2.5× el tamaño.

### 3. Para chat interactivo, `qwen2.5-3b Q4_K_M` es el punto dulce

16.7 tok/s de generación (más rápido que la velocidad de lectura), 3.6 GB de RAM,
calidad 0.70. Es la opción más rápida que no está comprometida en calidad, y deja
casi toda una máquina de 30 GB libre para lo demás.

## La guía de decisión

| caso de uso | elección | por qué |
|---|---|---|
| Chat interactivo / asistente | **qwen2.5-3b `Q4_K_M`** | 16 tok/s, 3.6 GB, calidad 0.70 |
| Mejor calidad, sobra RAM y se tolera ~7 tok/s | **qwen2.5-7b `Q3_K_M`** | 0.87 de calidad en 5.5 GB |
| Solo salida estructurada / restringida (JSON, una palabra) | **gemma-2-2b `Q4_K_M`** | 1.00 en seguir formato, rápido, chico — pero 0.44 en razonamiento |
| Punto medio | **phi-3.5-mini `Q4_K_M`** | 0.77 de calidad, 13.7 tok/s, 5.4 GB |
| — | *evitar* llama-3.1-8b, mistral-7b | sin ganancia de calidad sobre 3 B al doble del costo |

Reglas prácticas de los datos:
- **Default a `Q4_K_M`.** Bajar a `Q3_K_M` solo cuando la RAM apriete *y* hayas
  verificado la calidad con tus propios prompts.
- **7 B más o menos parte a la mitad la velocidad de 3 B** en CPU (~9 vs ~17 tok/s).
- **Por debajo de ~10 tok/s no es un chatbot en vivo** — es un worker de batch / agentes.

## Resultados completos

`results/cpu_all.json`, tabla en `RESULTS.md`, gráfico de Pareto `results/cpu_all.png`.

## Advertencias

Prueba de calidad de 30 ítems (direccional, no es un benchmark real). Solo
peticiones de a una — la curva de concurrencia (2–4 en paralelo, la pregunta real
de una máquina compartida) es la próxima sección. Los números de prefill/decode
usan los contadores de rendimiento propios de llama.cpp.

## GPU (Tesla P100) — the break-even

Same models on a Kaggle P100: decode is **3.5–4× faster** than CPU (3B ~60 tok/s,
7–8B ~35 tok/s) and **prefill is 20–30× faster** (~800–1700 vs ~55 tok/s). But a
GPU rents for ~8× a CPU box. So **on pure cost per token, the CPU box wins.**

Get the GPU when: (1) a person is waiting on the output — 60 tok/s vs 17 is a
different product; (2) you need a model that won't run on CPU — `qwen2.5-14b Q4`
hits **0.93** quality at a usable 19 tok/s on the P100, vs ~3 tok/s (unusable) on
CPU; (3) RAG / long prompts — prefill dominates and the GPU is 20–30× ahead.
On GPU, `Q8_0` is fine — the "Q4 only" rule is CPU-specific.
