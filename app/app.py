"""
Clasificador de Productos / Product Classifier -- interfaz Gradio.

Este archivo es solo interfaz: toda la logica de clasificacion vive en
clasificador.py (pipeline TF-IDF + LinearSVC con fallback a reglas de
keywords) y la de los datos de ejemplo en ejemplos.py.

Origen del modelo: product_classifier.ipynb, en la raiz del repo
(https://github.com/AgusPluda/product-classifier-project). Este Space lo
empaqueta como app usable; no reentrena nada distinto de lo que ya hace
ese notebook.
"""

from __future__ import annotations

import os

import gradio as gr

from clasificador import GLOSA_EN, clasificar
from ejemplos import (
    CATEGORIAS_TABLA,
    categoria_desde_opcion,
    ejemplos_por_categoria,
    opciones_dropdown,
)

EJEMPLOS_RAPIDOS = [
    "pencil",
    "red heart shape doormat",
    "mini playing cards spaceboy",
    "birthday card",
    "hot water bottle",
]


def _formatear_resultado(texto: str) -> str:
    resultado = clasificar(texto)

    if resultado["error"]:
        return f"⚠️ {resultado['mensaje']}"

    categoria = resultado["categoria"]
    glosa = GLOSA_EN.get(categoria, "")
    decidido_por = resultado["decidido_por"]
    margen = resultado["margen"]

    if decidido_por == "modelo":
        origen = f"Decidió: modelo ML · margen de confianza {margen:.2f} (≥ 0.20)"
        origen += "\nDecided by: ML model · confidence margin"
    elif decidido_por == "regla":
        origen = f"Decidió: regla de keywords (margen del modelo {margen:.2f} < 0.20)"
        origen += "\nDecided by: keyword rule (model margin below threshold)"
    else:
        origen = f"Sin regla ni margen suficiente (margen {margen:.2f} < 0.20)"
        origen += "\nNo rule matched and the model margin was too low"

    lineas = [
        f"## {categoria}",
        f"*{glosa}*" if glosa else "",
        "",
        origen,
        "",
        "**Top-3 candidatas del modelo / model's top-3** "
        "_(puntaje de decision_function, no es una probabilidad — "
        "score, not a probability)_",
    ]
    for cat, score in resultado["top3"]:
        marca = " ← elegida / chosen" if cat == categoria and decidido_por == "modelo" else ""
        lineas.append(f"- {cat} ({GLOSA_EN.get(cat, '')}): {score:.2f}{marca}")

    if resultado["categoria_regla"]:
        lineas += ["", f"Regla de keywords opinaba / keyword rule suggested: **{resultado['categoria_regla']}**"]

    return "\n".join(l for l in lineas if l is not None)


def _cargar_tabla(opcion_categoria: str):
    categoria = categoria_desde_opcion(opcion_categoria)
    return ejemplos_por_categoria(categoria)


def _clic_en_fila(evt: gr.SelectData, tabla: object):
    if evt.index is None:
        return gr.update()
    fila = evt.index[0]
    try:
        producto = tabla.iloc[fila, 0]
    except Exception:
        return gr.update()
    return producto


CSS_EXTRA = """
.categoria-resultado { min-height: 220px; }
"""

with gr.Blocks(title="Clasificador de Productos / Product Classifier") as demo:
    gr.Markdown(
        "# 🏷️ Clasificador de Productos / Product Classifier\n"
        "Escribí el nombre de un producto **en inglés** y el modelo va a decir a qué categoría "
        "de la taxonomía pertenece. *Type a product name in **English** and the model will "
        "predict its category.*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            entrada = gr.Textbox(
                label="Producto / Product",
                placeholder="pencil",
                autofocus=True,
            )
            boton = gr.Button("Clasificar / Classify", variant="primary")
            gr.Examples(examples=EJEMPLOS_RAPIDOS, inputs=entrada, label="Ejemplos rápidos / Quick examples")
            resultado = gr.Markdown(elem_classes=["categoria-resultado"])

        with gr.Column(scale=1):
            gr.Markdown("### Ejemplos por categoría / Examples by category")
            selector = gr.Dropdown(
                choices=opciones_dropdown(),
                value=opciones_dropdown()[0],
                label="Categoría / Category",
            )
            tabla = gr.Dataframe(
                value=ejemplos_por_categoria(CATEGORIAS_TABLA[0]),
                interactive=False,
                wrap=True,
            )
            gr.Markdown(
                "_Clic en una fila para probarla / Click a row to try it._ "
                "✓ = etiquetado a mano / hand-labeled."
            )

    with gr.Accordion("Cómo funciona / How it works", open=False):
        gr.Markdown(
            "**ES** — La taxonomía (17 categorías) y las etiquetas de entrenamiento salen de "
            "reglas de keywords (weak supervision), porque el dataset original no traía "
            "categoría de producto. Sobre esas etiquetas se entrena un pipeline "
            "TF-IDF (palabras + n-gramas de caracteres) + LinearSVC. En cada predicción, si "
            "el modelo no está lo bastante seguro (margen de confianza < 0.20), se usa la "
            "regla de keywords como respaldo. Contra 400 productos etiquetados a mano a "
            "ciegas, el modelo solo alcanza 78,0% de accuracy (F1 macro 0,746); la categoría "
            "más débil es *Decoración del Hogar* (catch-all, F1 ≈ 0,29), y la etiqueta "
            "*Otros* del gold no es predecible por diseño. Código completo del pipeline: "
            "[product_classifier.ipynb en GitHub]"
            "(https://github.com/AgusPluda/product-classifier-project).\n\n"
            "**EN** — The 17-category taxonomy and training labels come from keyword rules "
            "(weak supervision), since the source dataset had no product category. A "
            "TF-IDF (word + character n-grams) + LinearSVC pipeline is trained on those "
            "labels. At prediction time, if the model isn't confident enough (confidence "
            "margin < 0.20), the keyword rule is used as a fallback. Against 400 blind, "
            "hand-labeled products, the model alone reaches 78.0% accuracy (macro F1 0.746); "
            "the weakest category is *Home Decor* (a catch-all, F1 ≈ 0.29), and the gold "
            "*Other* label is unpredictable by design."
        )

    boton.click(_formatear_resultado, inputs=entrada, outputs=resultado)
    entrada.submit(_formatear_resultado, inputs=entrada, outputs=resultado)
    selector.change(_cargar_tabla, inputs=selector, outputs=tabla)
    tabla.select(_clic_en_fila, inputs=tabla, outputs=entrada)

if __name__ == "__main__":
    # server_name="0.0.0.0" y el puerto via $PORT son lo que Koyeb (y la
    # mayoria de los PaaS con capa gratuita) esperan para exponer la app;
    # 7860 es el default de Gradio, para correrlo en local sin variables.
    demo.launch(
        theme=gr.themes.Ocean(),
        css=CSS_EXTRA,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
