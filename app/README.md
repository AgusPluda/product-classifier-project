# Clasificador de Productos

🔗 **Demo en vivo:** https://product-classifier-project.onrender.com

(Corre en el free tier de Render: si no tuvo tráfico en un rato, el primer
request puede tardar unos segundos en "despertar" el servicio.)

Demo interactiva del modelo desarrollado en
[product_classifier.ipynb](https://github.com/AgusPluda/product-classifier-project/blob/main/product_classifier.ipynb):
TF-IDF (palabras + n-gramas de caracteres) + LinearSVC, con fallback a
reglas de keywords cuando la confianza del modelo es baja (margen < 0.20).

Escribí el nombre de un producto **en inglés** (el dataset original,
[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii),
es de una tienda mayorista de regalería con base en Reino Unido) y la app
predice a cuál de las 17 categorías de la taxonomía pertenece. Incluye una
tabla de 25 ejemplos por categoría para saber qué tipo de texto ingresar.

No se versiona ningún modelo serializado: el pipeline entrena al arrancar
la app (menos de 1 segundo sobre ~2.880 ejemplos), así que siempre corre
con la versión de scikit-learn fijada en `requirements.txt`.

## Archivos

- `app.py` — interfaz Gradio (tema Ocean).
- `clasificador.py` — normalización de texto, reglas de keywords,
  entrenamiento del pipeline y la función `clasificar()`.
- `ejemplos.py` — arma la tabla de ejemplos por categoría.
- `data/` — copia de `data/processed/products_categorized.csv` y
  `data/processed/gold_labels.csv` del repo principal.
- `Dockerfile` — imagen usada por Render para levantar el servicio (build
  context = esta carpeta).

## Re-sincronizar los datos

Si el catálogo o el gold set cambian en el repo principal, actualizar las
copias de acá:

```bash
cp ../data/processed/products_categorized.csv data/
cp ../data/processed/gold_labels.csv data/
```

## Correr localmente

```bash
pip install -r requirements.txt
python app.py
```

## Deploy

Servicio en Render (free tier), conectado a este repo con **Root
Directory = `app`** y builder **Docker**. Un push a `main` no redeploya
solo (el repo está conectado como "Public Git Repository", sin
auto-deploy) — hay que entrar al dashboard de Render y usar **Manual
Deploy → Deploy latest commit**.

## Repo del proyecto

https://github.com/AgusPluda/product-classifier-project
