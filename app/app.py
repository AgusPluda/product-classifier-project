"""
Clasificador de Productos -- interfaz Gradio.

Este archivo es solo interfaz: toda la logica de clasificacion vive en
clasificador.py (pipeline TF-IDF + LinearSVC con fallback a reglas de
keywords) y la de los datos de ejemplo en ejemplos.py.

Origen del modelo: product_classifier.ipynb, en la raiz del repo
(https://github.com/AgusPluda/product-classifier-project). Este deploy lo
empaqueta como app usable; no reentrena nada distinto de lo que ya hace
ese notebook.
"""

from __future__ import annotations

import os

import gradio as gr
import plotly.graph_objects as go
from clasificador import UMBRAL_CONFIANZA, clasificar
from ejemplos import CATEGORIAS_TABLA, ejemplos_por_categoria

COLOR_ELEGIDA = "#1a9c8c"  # acorde a la paleta del tema Ocean
COLOR_CANDIDATA = "#5b6b76"
COLOR_LINEA_UMBRAL = "#e07b39"

EJEMPLOS_RAPIDOS = [
    "pencil",
    "red heart shape doormat",
    "mini playing cards spaceboy",
    "birthday card",
    "hot water bottle",
]


ALTURA_GRAFICO = 375


def _layout_grafico(fig: go.Figure, **kwargs) -> go.Figure:
    """Layout comun a todos los estados del grafico (vacio, con datos),
    para que el "piso" no salte entre el estado inicial (antes de
    clasificar) y el estado con resultados.
    """
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=ALTURA_GRAFICO,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#9aa5ad",
        showlegend=False,
        **kwargs,
    )
    return fig


def _grafico_vacio() -> go.Figure:
    """Placeholder con la misma altura que el grafico real, para que el
    bloque no cambie de tamaño cuando aparece el primer resultado.
    """
    fig = go.Figure()
    return _layout_grafico(fig, xaxis={"visible": False}, yaxis={"visible": False})


def _grafico_top3(resultado: dict) -> go.Figure:
    """Barras horizontales con el top-3 del modelo (decision_function, no
    es una probabilidad) y una linea vertical en el umbral que el top-1
    necesitaba superar para que el modelo decidiera (score del 2do + 0.20).
    """
    top3 = list(reversed(resultado["top3"]))  # 1er lugar arriba del grafico
    categorias = [cat for cat, _ in top3]
    scores = [score for _, score in top3]
    colores = [
        COLOR_ELEGIDA
        if (cat == resultado["categoria"] and resultado["decidido_por"] == "modelo")
        else COLOR_CANDIDATA
        for cat in categorias
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=categorias,
            orientation="h",
            marker_color=colores,
            text=[f"{s:.2f}" for s in scores],
            textposition="outside",
        )
    )

    segundo_score = resultado["top3"][1][1]
    umbral_abs = segundo_score + UMBRAL_CONFIANZA
    fig.add_vline(
        x=umbral_abs,
        line_dash="dash",
        line_color=COLOR_LINEA_UMBRAL,
        annotation_text="margen mínimo para que decida el modelo",
        annotation_position="top",
        annotation_font_color=COLOR_LINEA_UMBRAL,
    )

    return _layout_grafico(fig, xaxis_title="puntaje (decision_function)")


def _formatear_resultado(texto: str):
    resultado = clasificar(texto)

    if resultado["error"]:
        return f"⚠️ {resultado['mensaje']}", _grafico_vacio()

    categoria = resultado["categoria"]
    decidido_por = resultado["decidido_por"]
    margen = resultado["margen"]

    if decidido_por == "modelo":
        origen = f"Decidió: modelo ML · margen de confianza {margen:.2f} (≥ 0.20)"
    elif decidido_por == "regla":
        origen = f"Decidió: regla de keywords (margen del modelo {margen:.2f} < 0.20)"
    else:
        origen = f"Sin regla ni margen suficiente (margen {margen:.2f} < 0.20)"

    lineas = [f"# {categoria}", "", origen]
    if resultado["categoria_regla"]:
        lineas += ["", f"Regla de keywords opinaba: **{resultado['categoria_regla']}**"]

    return "\n".join(lineas), _grafico_top3(resultado)


def _cargar_tabla(categoria: str):
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
/* height fija (no min-height): asi el bloque mide lo mismo vacio que
   poblado, y no empuja al grafico de abajo cuando aparece el resultado.
   overflow visible (no hidden) para no cortar la linea de la regla en
   categorias con nombre largo que lleguen a ocupar 2 lineas. */
.categoria-resultado { height: 190px; overflow: visible; font-size: 1.9rem; line-height: 1.85; }
/* El Dropdown de Gradio renderea mas alto que el Textbox por defecto;
   se fuerza a ambos "bloques" (Producto / Categoria) a la misma altura,
   tomando como referencia la caja de Producto. */
.campo-igualado { min-height: 92px; box-sizing: border-box; }
.grafico-top3 { margin-top: -76px; }
"""

with gr.Blocks(title="Clasificador de Productos") as demo:
    gr.Markdown(
        "# 🏷️ Clasificador de Productos\n"
        "Modelo entrenado sobre ventas reales de una tienda mayorista de *regalería* "
        "(regalos, decoración y bazar), así que funciona mejor con ese tipo de productos — "
        "probá con los ejemplos de abajo si no sabés por dónde arrancar.\n\n"
        "Escribí el nombre de un producto **en inglés** y el modelo va a decir a qué "
        "categoría de la taxonomía pertenece."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Predecir")
            entrada = gr.Textbox(
                label="Producto",
                placeholder="pencil",
                autofocus=True,
                elem_classes=["campo-igualado"],
            )
            boton = gr.Button("Clasificar", variant="primary")
            gr.Examples(
                examples=EJEMPLOS_RAPIDOS, inputs=entrada, label="Ejemplos rápidos"
            )
            resultado = gr.Markdown(elem_classes=["categoria-resultado"])
            grafico = gr.Plot(
                value=_grafico_vacio(),
                label="Top-3 candidatas del modelo",
                elem_classes=["grafico-top3"],
            )

        with gr.Column(scale=1):
            gr.Markdown("### Ejemplos por categoría")
            selector = gr.Dropdown(
                choices=CATEGORIAS_TABLA,
                value=CATEGORIAS_TABLA[0],
                label="Categoría",
                elem_classes=["campo-igualado"],
            )
            gr.Markdown("_Clic en una fila para probarla._ ✓ = etiquetado a mano.")
            tabla = gr.Dataframe(
                value=ejemplos_por_categoria(CATEGORIAS_TABLA[0]),
                interactive=False,
                wrap=True,
                max_height=560,
            )

    with gr.Accordion("Cómo funciona", open=False):
        gr.Markdown(
            "El modelo entrena sobre [Online Retail II]"
            "(https://archive.ics.uci.edu/dataset/502/online+retail+ii), las transacciones "
            "2009-2011 de un mayorista de regalería con base en Reino Unido: velas, "
            "papelería, decoración del hogar, bijouterie y similares. Fuera de ese rubro "
            "(electrónica, indumentaria, alimentos, etc.) el modelo no tiene con qué "
            "comparar y probablemente prediga cualquier cosa.\n\n"
            "La taxonomía (17 categorías) y las etiquetas de entrenamiento salen de "
            "reglas de keywords (weak supervision), porque el dataset original no traía "
            "categoría de producto. Sobre esas etiquetas se entrena un pipeline "
            "TF-IDF (palabras + n-gramas de caracteres) + LinearSVC. En cada predicción, si "
            "el modelo no está lo bastante seguro (margen de confianza < 0.20), se usa la "
            "regla de keywords como respaldo. Contra 400 productos etiquetados a mano a "
            "ciegas, el modelo solo alcanza 78,0% de accuracy (F1 macro 0,746); la categoría "
            "más débil es *Decoración del Hogar* (catch-all, F1 ≈ 0,29), y la etiqueta "
            "*Otros* del gold no es predecible por diseño. Código completo del pipeline: "
            "[product_classifier.ipynb en GitHub]"
            "(https://github.com/AgusPluda/product-classifier-project)."
        )

    boton.click(_formatear_resultado, inputs=entrada, outputs=[resultado, grafico])
    entrada.submit(_formatear_resultado, inputs=entrada, outputs=[resultado, grafico])
    selector.change(_cargar_tabla, inputs=selector, outputs=tabla)
    tabla.select(_clic_en_fila, inputs=tabla, outputs=entrada)

if __name__ == "__main__":
    # server_name="0.0.0.0" y el puerto via $PORT son lo que Render (y la
    # mayoria de los PaaS con capa gratuita) esperan para exponer la app;
    # 7860 es el default de Gradio, para correrlo en local sin variables.
    demo.launch(
        theme=gr.themes.Ocean(),
        css=CSS_EXTRA,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
