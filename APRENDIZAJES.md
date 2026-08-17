# Aprendizajes del proyecto: Clasificador de Productos

Este documento resume el recorrido conceptual del proyecto — no el código en sí (eso está en
el notebook), sino el *porqué* de cada decisión y las herramientas de scikit-learn que fuimos
usando en el camino. La idea es que sirva como referencia para entender cualquier parte del
pipeline sin tener que releer todo el notebook, y como base para replicar el enfoque en otros
proyectos de clasificación de texto.

---

## 1. El problema de fondo: no había etiquetas

Todo el diseño del proyecto parte de una restricción central: el dataset tiene descripciones
de producto, pero **ninguna columna de categoría comercial**. Esto descarta de entrada cualquier
enfoque de "clasificación supervisada" ingenuo (no hay `y` para entrenar un modelo) y obliga a
construir las etiquetas antes de poder construir el clasificador.

La técnica para resolver esto se llama **weak supervision** (supervisión débil): en vez de
etiquetar miles de productos a mano, se generan etiquetas aproximadas y ruidosas con reglas
heurísticas baratas (en este caso, keywords/regex), se entrena un modelo sobre esas etiquetas
imperfectas, y se mide la calidad real del sistema contra una muestra chica etiquetada a mano
con cuidado (el "gold set"). El valor del modelo no es imitar las reglas — es **generalizar
más allá** de lo que las reglas pudieron capturar.

---

## 2. El pipeline completo, de un vistazo

```
Catálogo de productos          (deduplicar transacciones -> un producto por stock_code)
        |
Clustering exploratorio        (K-Means sobre TF-IDF, para descubrir que categorias existen)
        |
Taxonomía                      (15 categorias definidas a mano, con ejemplo del cluster)
        |
Reglas por keywords            (weak labels: etiqueta aproximada para ~70% del catalogo)
        |
Set gold (~400, manual)        (unica fuente de verdad confiable, etiquetado a ciegas)
        |
Modelo supervisado (SVM)       (entrenado con las reglas, EVALUADO contra el gold)
        |
Inferencia + calibración       (aplicar a todo el catalogo, con umbral de confianza)
        |
Análisis de negocio            (ventas por categoria, estacionalidad, etc.)
```

Cada flecha de este diagrama representa una capa que **compensa la debilidad de la anterior**:
el clustering no etiqueta nada pero revela qué categorías existen; las reglas etiquetan rápido
pero con ruido; el gold set es lento de producir pero es la única verdad confiable; el modelo
aprende de las reglas pero se evalúa contra el gold; la calibración de confianza corrige el
sesgo que el modelo comete cuando no sabe qué responder.

---

## 3. Concepto por concepto, en el orden en que aparecieron

### 3.1 Construcción del catálogo

El dataset de transacciones tenía 1.003.417 filas pero solo ~4.700 productos únicos. El primer
error a evitar fue deduplicar mal: un mismo `stock_code` aparecía con 2 a 4 descripciones
distintas por errores de tipeo (`RETRO LEAVES MAGNETIC NOTEPAD` vs. `RETO LEAVES MAGNETIC
SHOPPING LIST`). La solución no fue quedarse con la primera que aparece ni con la más "prolija",
sino con **la que más veces se repite en transacciones reales** — la frecuencia de uso real es
una señal más confiable que el orden de aparición para decidir cuál es la versión "correcta".

**Aprendizaje general:** cuando hay que elegir un valor canónico entre variantes ruidosas de un
mismo dato, la frecuencia de uso real suele ser mejor criterio que la primera ocurrencia o un
criterio estético.

### 3.2 Clustering exploratorio

Objetivo: no clasificar nada todavía, sino **descubrir qué categorías tiene sentido definir**,
mirando cómo se agrupa el texto de las descripciones de forma no supervisada.

- **TF-IDF** (Term Frequency – Inverse Document Frequency) convierte cada descripción en un
  vector numérico donde cada palabra (o n-grama) pesa más si aparece seguido *en ese producto*
  pero es rara *en el resto del catálogo*. Es lo que permite que "CHRISTMAS" o "CANDLE" pesen
  más que "SET" o "OF", que aparecen en todos lados y no aportan información distintiva.
- **K-Means** agrupa los vectores en `k` clusters minimizando la varianza dentro de cada grupo
  (la "inercia"). Hay que decidir `k` de antemano — no lo elige el algoritmo solo.
- **El método del codo casi no sirvió acá.** Con texto disperso de alta dimensión (miles de
  columnas, la mayoría en cero), la inercia cae casi en línea recta en vez de mostrar un quiebre
  claro — es un efecto conocido de la "maldición de la dimensionalidad", no un error nuestro.
- **Silhouette score** (qué tan bien separado y cohesionado está cada cluster) tampoco dio
  valores altos en términos absolutos (normal en texto disperso), pero sí sirvió como guía
  *relativa* para comparar distintos `k` entre sí.
- **El hallazgo más importante de esta etapa:** los primeros clusters se agrupaban por
  **color/material/estilo** (`PINK`, `BLUE`, `VINTAGE`, `GLASS`) en vez de por tipo de producto,
  porque esas palabras son las más frecuentes de todo el catálogo y dominaban la distancia entre
  vectores. La solución fue ampliar la lista de `stop_words` con esos modificadores para que el
  vectorizador se quedara con las palabras que sí indican *qué es* el producto.
- **Estrategia de "sobre-clusterizar y fusionar a mano"**: en vez de pedirle a K-Means que
  encuentre de una las 15 categorías finales, se usó un `k` mucho más alto (45) para obtener
  grupos finos y coherentes, y recién después se fusionaron a mano en la taxonomía final. Es
  mucho más fácil fusionar clusters parecidos que separar un cluster que quedó mezclado.

### 3.3 Definición de la taxonomía

Con los clusters como insumo, se definieron 15 categorías mutuamente excluyentes a mano. Regla
de diseño clave: la categoría `Otros` tiene que ser la excepción, no el destino por defecto —
si absorbe más del 15% del catálogo, es señal de que falta una categoría real en la taxonomía.

### 3.4 Reglas por keywords (weak labels)

Un diccionario `categoría -> [patrones regex]`, aplicado en **orden de prioridad** (la primera
categoría cuyo patrón matchea, gana). Esto generó una etiqueta aproximada para ~70% del catálogo
en segundos, sin etiquetar nada a mano todavía.

Errores reales que aparecieron acá y vale la pena recordar:

- **`\bpalabra\b` no matchea el plural.** `\bbowl\b` no encuentra `BOWLS`, porque no hay límite
  de palabra entre la `l` y la `s` (ambas son parte del mismo token). Hubo que revisar cada
  patrón y agregar `s?` donde correspondía.
- **Ambigüedad "empaque vs. producto".** `BOX OF 9 PEBBLE CANDLES` no es una caja, es un paquete
  de velas — pero un regex genérico de `\bbox\b` no puede distinguir cuándo una palabra es el
  producto y cuándo es solo el envase. Esto no tiene una solución perfecta con regex simple; es
  una limitación aceptada y documentada, no algo que valga la pena perseguir con reglas cada vez
  más frágiles.
- **Una palabra genérica puede "contaminar" el entrenamiento aunque la regla esté bien
  diseñada.** Se agregó `\bhearts?\b` a una categoría catch-all razonando "es seguro porque esta
  categoría se revisa última en el orden de prioridad" — y ese razonamiento era válido *para las
  reglas*, pero no protegía al modelo de ML entrenado después: el modelo no sabe nada de
  prioridades, solo ve que muchas filas con la palabra "heart" fueron etiquetadas con esa
  categoría, y aprende esa asociación aunque sea espuria (porque "heart" es un motivo/forma que
  aparece en joyería, cocina, velas, decoración por igual). **Lección:** una regla puede ser
  "segura" para resolver conflictos internos del diccionario y aun así ser una mala idea como
  feature de entrenamiento.

### 3.5 El set gold: la única fuente de verdad

Sin un conjunto de datos etiquetado por un humano, no hay forma honesta de saber si el modelo
funciona — solo se puede medir qué tan bien imita a las reglas, que es circular.

- **Muestreo estratificado por partes iguales**, no proporcional al tamaño real de cada
  categoría. Si se muestrea proporcional al catálogo, las categorías chicas (41 productos) caen
  a 3-4 ejemplos en el gold, insuficientes para medir su precisión/recall de forma confiable.
  Tomar la misma cantidad de cada categoría (incluida una porción del "sin regla") da métricas
  por clase estadísticamente más sólidas, a costa de no representar la proporción real de ventas
  — lo cual está bien, porque el gold set existe para *evaluar*, no para *reflejar* el catálogo.
- **Etiquetado a ciegas**: el archivo para completar a mano no incluía la predicción de la regla.
  Si la hubiera incluido, es muy fácil "anclarse" en lo que ya dice la regla en vez de juzgar el
  producto de manera independiente — eso invalidaría el propósito de tener una evaluación
  separada.
- **Medir el baseline de las reglas contra el gold antes de entrenar nada** — así se sabe de
  entrada qué número tiene que superar el modelo para que valga la pena.

### 3.6 El modelo supervisado

- **Excluir del entrenamiento los `stock_code` que están en el gold.** Es la misma lógica de
  fuga de datos (data leakage) de siempre: si el modelo ve esos productos durante el
  entrenamiento (aunque sea con una etiqueta débil), la evaluación final deja de ser honesta.
- **Comparar arquitecturas simples antes de afinar hiperparámetros**: un modelo *Dummy* (predice
  siempre la clase más frecuente) sirve de piso para confirmar que no hay nada raro pasando;
  Naive Bayes, Regresión Logística y SVM lineal se comparan con validación cruzada antes de
  invertir tiempo ajustando el que "parece" mejor a ojo.
- **`f1_macro` en vez de `accuracy`** como métrica de comparación: `accuracy` se deja dominar por
  las categorías grandes y esconde si el modelo abandona a las chicas; `f1_macro` promedia el F1
  de cada clase por igual, sin importar cuántos ejemplos tenga.
- **`class_weight="balanced"`** compensa parcialmente el desbalance de clases durante el
  entrenamiento, dándole más peso al error en las clases chicas — pero, como se vio en el
  Paso 7, no elimina del todo el sesgo hacia la clase mayoritaria cuando el modelo está inseguro.
- **El resultado de la validación cruzada (0.985 de f1_macro) no significa lo que parece
  significar.** Esa cifra mide qué tan bien el modelo reproduce las etiquetas de las reglas — y
  como esas etiquetas vienen de buscar las mismas palabras clave que el modelo puede "ver"
  directamente en el texto, es casi trivial que el ajuste sea casi perfecto. El número que
  importa de verdad es el que sale de evaluar contra el gold set (0.692 de f1_macro), porque ahí
  sí se mide generalización real, no memorización de las reglas.

### 3.7 Inferencia sobre el catálogo completo y calibración de confianza

- **Un modelo lineal uno-contra-todos tiende a "adivinar" la clase más grande cuando está
  inseguro.** Se vio con evidencia concreta: en los 50 productos de menor confianza del catálogo
  completo, más de la mitad cayeron en la categoría más numerosa del entrenamiento (`Cocina y
  Mesa`), sin relación real con esos productos (`SILVER TEDDY BEAR`, `GOLD STANDING GNOME`). Es
  un efecto de que la clase con más datos ocupa más espacio de decisión en un modelo lineal,
  incluso con `class_weight="balanced"`.
- **No asumir un umbral de confianza — calibrarlo con datos reales.** En vez de elegir un número
  "razonable" a ojo, se usó el gold set (donde sabemos si el modelo acertó o no) para medir el
  accuracy por debajo y por encima de distintos umbrales de margen de confianza, y se eligió el
  punto donde la separación entre "acierta poco" y "acierta mucho" es más clara.
- **Abstención en vez de forzar una respuesta**: para las predicciones de baja confianza, en vez
  de confiar ciegamente en el modelo, se usa la regla como respaldo (si existe) o se marca
  explícitamente `"Sin clasificar"`. El accuracy total no mejoró por hacer esto (71.5% vs. 71.8%,
  diferencia dentro del margen de ruido) — pero sí corrigió el sesgo sistemático hacia la clase
  grande, que es justamente lo que le daría números falsos al análisis de ventas del paso
  siguiente. Es un ejemplo de que una métrica global puede quedarse igual mientras se arregla un
  problema real que esa métrica no estaba midiendo.

---

## 4. Glosario de funciones de scikit-learn usadas

| Función / clase | Módulo | Qué hace | Por qué la usamos acá |
|---|---|---|---|
| `TfidfVectorizer` | `sklearn.feature_extraction.text` | Convierte texto en una matriz numérica donde cada columna es una palabra (o n-grama) y cada celda pesa más si el término es frecuente en ese documento pero raro en el resto del corpus. | Base de todo el proyecto: transforma las descripciones de producto en vectores que un modelo puede procesar. Se usó tanto para el clustering (Paso 2) como para el modelo final (Paso 6), con distintos parámetros. |
| `ENGLISH_STOP_WORDS` | `sklearn.feature_extraction.text` | Lista predefinida de palabras inglesas sin contenido semántico propio (`the`, `of`, `and`...) que se excluyen del vocabulario. | Se combinó con una lista propia de modificadores (colores, materiales, tamaños) para evitar que dominaran el clustering por sobre las palabras que indican tipo de producto. |
| `KMeans` | `sklearn.cluster` | Agrupa vectores en `k` clusters, minimizando la distancia de cada punto al centro de su grupo. | Clustering exploratorio para descubrir qué categorías de producto existen naturalmente en el catálogo, antes de definir la taxonomía a mano. |
| `silhouette_score` | `sklearn.metrics` | Mide qué tan bien separado y compacto está cada cluster, en una escala de -1 a 1. | Guía relativa (no absoluta) para comparar distintos valores de `k` en el clustering exploratorio. |
| `DummyClassifier` | `sklearn.dummy` | Un "modelo" que ignora el texto y predice siempre según una regla trivial (ej. la clase más frecuente). | Piso de referencia: si un modelo real no le gana claramente al Dummy, algo anda mal en el pipeline. |
| `MultinomialNB` | `sklearn.naive_bayes` | Clasificador probabilístico (Naive Bayes) pensado para conteos/frecuencias de features, como las de TF-IDF. | Uno de los cuatro modelos comparados en la validación cruzada inicial. |
| `LogisticRegression` | `sklearn.linear_model` | Clasificador lineal que estima la probabilidad de cada clase mediante una función logística. | Comparado en la validación cruzada; buen desempeño pero superado por SVM lineal. |
| `LinearSVC` | `sklearn.svm` | Support Vector Machine con kernel lineal: busca el hiperplano que mejor separa las clases maximizando el margen entre ellas. | El modelo ganador de la comparación (f1_macro más alto). No tiene `predict_proba`, pero sí `decision_function`, usada como proxy de confianza en el Paso 7. |
| `Pipeline` | `sklearn.pipeline` | Encadena pasos de transformación y modelado en un solo objeto, para que se ajusten juntos y sin fugas de datos entre entrenamiento y validación. | Combina el vectorizador TF-IDF con el clasificador en un solo flujo, reutilizable en `cross_val_score` y `GridSearchCV`. |
| `FeatureUnion` | `sklearn.pipeline` | Aplica varios transformadores en paralelo sobre el mismo input y concatena sus salidas en una sola matriz de features. | Combina un `TfidfVectorizer` de palabras (1-2 gramas) con uno de caracteres (3-5 gramas) — el de caracteres ayuda a capturar variantes morfológicas y errores de tipeo que el de palabras solo no detecta. |
| `StratifiedKFold` | `sklearn.model_selection` | Divide los datos en `k` particiones (folds) para validación cruzada, preservando la proporción de cada clase en cada partición. | Evita que una partición se quede sin ejemplos de una categoría chica, lo cual rompería la métrica en ese fold. |
| `cross_val_score` | `sklearn.model_selection` | Entrena y evalúa un modelo en cada fold de una validación cruzada, devolviendo un score por fold. | Comparación inicial de las cuatro arquitecturas de modelo (Dummy, Naive Bayes, Regresión Logística, SVM). |
| `GridSearchCV` | `sklearn.model_selection` | Prueba combinaciones de hiperparámetros mediante validación cruzada y se queda con la que mejor score obtiene. | Ajuste del parámetro `C` (regularización) del SVM lineal ganador. |
| `classification_report` | `sklearn.metrics` | Genera un reporte de precision, recall y F1-score por clase, más promedios macro y ponderado. | Diagnóstico detallado del modelo final contra el gold set — permitió detectar que `Decoración del Hogar` y `Papelería y Tarjetería` eran las categorías más débiles. |
| `f1_score` | `sklearn.metrics` | Calcula el F1-score (media armónica de precision y recall), con distintas formas de promediar entre clases (`macro`, `weighted`, etc.). | Métrica principal de comparación entre el modelo y el baseline de reglas. |
| `accuracy_score` | `sklearn.metrics` | Proporción de predicciones exactamente correctas sobre el total. | Métrica complementaria al f1_macro, más intuitiva para comunicar el resultado ("acierta 7 de cada 10"). |
| `ConfusionMatrixDisplay` | `sklearn.metrics` | Grafica una matriz de confusión (qué clases se confunden entre sí) a partir de predicciones y valores reales. | Visualización de los errores del modelo final contra el gold set. |

---

## 5. Errores prácticos que vale la pena recordar (no son de Machine Learning, pero costaron tiempo)

- **`pandas` 3.0 cambió el comportamiento de `groupby().apply()`**: ya no incluye la columna de
  agrupamiento dentro de lo que le pasa a la función aplicada. El método `GroupBy.sample()`
  (nativo, sin pasar por `apply`) resuelve el mismo problema de muestreo estratificado sin este
  inconveniente.
- **Excel en Windows no exporta CSV en UTF-8 por default** — usa `cp1252` (ANSI), lo que rompe
  la lectura de tildes si `pandas.read_csv` no especifica el encoding correcto.
- **La configuración regional en español hace que Excel use `;` como separador de CSV**, no
  `,` (porque la coma está reservada para separar decimales). Si el separador no coincide con lo
  que espera `read_csv`, todas las columnas se leen mal sin que salte un error obvio.
- **Excel puede agregar espacios de relleno al alinear columnas**, lo que deja nombres de columna
  y valores con espacios invisibles al abrir el archivo — `skipinitialspace=True` y
  `.str.strip()` en columnas y valores lo resuelven.
- **Una lista de categorías separada por comas es ambigua si alguna categoría tiene una coma en
  el nombre propio** (`Espejos, Relojes y Arte de Pared`). Esto causó que la categoría se partiera
  en dos durante el etiquetado manual del gold set — la lección es evitar comas dentro de nombres
  de categoría, o usar un separador distinto (saltos de línea, por ejemplo) al comunicar listas
  de opciones para completar a mano.

---

## 6. Principios generales, reutilizables en otros proyectos

1. **Si no hay etiquetas, no hay clasificación supervisada posible sin antes construirlas** —
   ya sea a mano, con weak supervision, o con una combinación de ambas.
2. **Nunca evalúes un modelo con las mismas etiquetas ruidosas con las que lo entrenaste.** La
   validación cruzada sobre datos de entrenamiento generados por reglas mide "qué tan bien
   aprendió las reglas", no "qué tan bueno es el modelo".
3. **Una regla puede ser correcta en su contexto (resolver un conflicto de prioridad) y aun así
   ser una mala feature de entrenamiento** si es demasiado genérica.
4. **El "accuracy" global puede esconder problemas reales.** Un cambio que no mueve el accuracy
   puede estar corrigiendo un sesgo sistemático importante — hay que mirar la distribución de
   predicciones, no solo el promedio.
5. **Calibrar umbrales con datos reales, no a ojo.** Si existe aunque sea una muestra chica
   etiquetada a mano, usarla para medir en qué punto una señal de confianza se vuelve confiable,
   en vez de adivinar un número redondo.
6. **Diseñar para poder abstenerse.** Un modelo que siempre tiene que dar una respuesta, aunque
   no tenga ninguna señal real, va a "adivinar" de forma sesgada. Es mejor decir explícitamente
   "no sé" que forzar una predicción de baja confianza en un análisis de negocio.
