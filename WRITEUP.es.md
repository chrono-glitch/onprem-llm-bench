# Servir LLMs on-premise sin GPU: los números y cómo elegir

## El problema real

Una empresa quiere un LLM corriendo en su propio hardware — los datos no pueden
salir del edificio, o una cuenta en la nube por token no le cierra. Tienen un
servidor normal: 8 núcleos de CPU, ~30 GB de RAM, **sin GPU**. ¿Qué modelo
abierto, a qué cuantización, en qué motor — realmente funciona ahí, y alcanza
para producción? ¿Y vale la pena comprar una GPU?

Probé 7 modelos instruct (2–8 B) a 2–3 niveles de cuantización GGUF cada uno, en
exactamente esa máquina (8 vCPU AMD EPYC 7B12, `llama-cpp-python`), midiendo
velocidad de generación, RAM, tiempo de carga, concurrencia, ruido entre
corridas, dos motores (`llama-cpp-python` vs Ollama), la comparación contra una
GPU (Tesla P100 de Kaggle), y una prueba de calidad de 30
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

## GPU (Tesla P100) — el punto de equilibrio

Los mismos modelos en una P100 de Kaggle: el decode es **3.5–4× más rápido** que
en CPU (3B ~60 tok/s, 7–8B ~35 tok/s) y el **prefill es 20–30× más rápido**
(~800–1700 vs ~55 tok/s). Pero una GPU se renta a ~8× el costo de una caja de
CPU. Así que **en costo puro por token, gana la caja de CPU.**

Conseguí la GPU cuando: (1) hay una persona esperando el output — 60 tok/s vs 17
es otro producto; (2) necesitás un modelo que no corre en CPU — `qwen2.5-14b Q4`
da **0.93** de calidad a 19 tok/s usables en la P100, vs ~3 tok/s (inusable) en
CPU; (3) RAG / prompts largos — domina el prefill y la GPU va 20–30× adelante.
En GPU, `Q8_0` está bien — la regla "solo Q4" es específica de CPU.

## Concurrencia: una caja de CPU es una unidad de un solo usuario

Primero, un hallazgo de infraestructura: un objeto `Llama` de `llama-cpp-python`
**no es seguro para concurrencia** — dos hilos llamándolo al mismo tiempo hacen
segfault. On-prem hay que ponerle un servidor adelante (`llama_cpp.server`), que
encola las peticiones.

Con ese servidor y `qwen2.5-3b Q4_K_M`, disparando C peticiones en paralelo:

| concurrencia | latencia p50 | p95 | tok/s agregado | latencia vs C=1 |
|---|---|---|---|---|
| 1 | 5.7 s | 5.7 s | 16.4 | 1.0× |
| 2 | 7.8 s | 10.5 s | 18.0 | 1.4× |
| 4 | 13.0 s | 20.9 s | 18.0 | 2.3× |
| 8 | 27.6 s | 45.8 s | 16.4 | 4.9× |

**El throughput agregado es plano (~16–18 tok/s) sin importar la concurrencia** —
una sola petición ya satura los 8 núcleos, así que las peticiones paralelas solo
hacen cola. La latencia crece casi lineal con la carga. Una caja de CPU sirve
**un usuario interactivo a la vez**, o una cola de batch que nadie espera en vivo.
Para N usuarios concurrentes: ~N cajas, o una GPU.

## Motor: Ollama vs llama-cpp-python — el wrapper es gratis

Los dos son llama.cpp por debajo. Importé los mismos archivos GGUF a Ollama y
corrí la misma prueba. **Ratio de decode promedio: 1.00×** — la capa HTTP y el
scheduler de Ollama no cuestan nada en throughput. El prefill hasta es un poco
más rápido en Ollama (sus defaults activan flash-attention). Calidad idéntica.

Entonces la elección es de operación, no de velocidad. Ollama te da gestión de
modelos (`pull`/`tag`/`rm`), un endpoint compatible con OpenAI siempre activo,
carga/descarga automática, y una cola de peticiones — sin penalización de tok/s.
**Para servir on-prem, usá Ollama o `llama_cpp.server`.** `llama-cpp-python`
en proceso solo es más simple para un benchmark de una sola pasada. (Ollama copia
cada GGUF a `~/.ollama`, ~2 GB por modelo.)

## Un hallazgo más: una caja compartida te cuesta previsibilidad

Volví a medir algunas celdas 5× cada una. `qwen2.5-3b Q4` es estable (±1.5 %);
los modelos de 7–8B varían ±10 % entre corridas, y *todos* cayeron ~40 % cuando
la caja tenía carga de otros procesos (load ~7 en 8 cores). Las cajas on-prem
casi siempre son compartidas — **dimensioná para ~0.6× el número de referencia y
±10 %**, y preferí el modelo que se degrada con gracia (`qwen2.5-3b`) si la caja
hace otras cosas.

## Advertencias

Prueba de calidad de 30 ítems, determinística a temp=0, direccional — un
termómetro que detecta un quant roto, no un ranking tipo MMLU. Los números de
prefill/decode usan los contadores de llama.cpp. El grid de CPU de referencia es
de un solo inquilino, una corrida por celda; ver la envolvente de ruido arriba.

---

*English version: [`WRITEUP.en.md`](WRITEUP.en.md).*
