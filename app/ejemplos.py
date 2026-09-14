"""
Construye la tabla de 25 productos de ejemplo por categoria, para que quien
usa la app sepa que tipo de texto ingresar (en ingles).

Prioridad de las fuentes, para que los ejemplos sean confiables y no
circulares (no queremos mostrar como "ejemplo" lo que el propio modelo
predijo sin verificacion humana):

  1. Primero, el gold set (`gold_labels.csv`, 400 productos etiquetados a
     mano a ciegas) -- se marcan como verificados.
  2. Si faltan para llegar a 25, se completa desde el catalogo completo
     (`products_categorized.csv`) con productos donde la regla de keywords
     y el modelo coinciden (`categoria_regla == categoria_predicha`), que
     es la senal de mayor confiabilidad disponible fuera del gold.

Se excluye la categoria "Otros": no es parte de la taxonomia que el
clasificador puede predecir (ver README del proyecto).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from clasificador import GLOSA_EN, ORDEN_PRIORIDAD

DATA_DIR = Path(__file__).parent / "data"
RUTA_CATALOGO = DATA_DIR / "products_categorized.csv"
RUTA_GOLD = DATA_DIR / "gold_labels.csv"

N_POR_CATEGORIA = 25
CATEGORIAS_TABLA = list(ORDEN_PRIORIDAD)  # excluye "Otros" y "Sin clasificar"

COL_PRODUCTO = "Producto / Product"
COL_VERIFICADO = "Verificado / Verified"


@lru_cache(maxsize=1)
def _cargar_fuentes() -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = pd.read_csv(RUTA_GOLD, encoding="utf-8")
    gold["categoria_gold"] = gold["categoria_gold"].str.strip()
    gold["description_clean"] = gold["description_clean"].str.strip()

    catalogo = pd.read_csv(RUTA_CATALOGO, encoding="utf-8")
    catalogo["description_clean"] = catalogo["description_clean"].str.strip()

    return gold, catalogo


def ejemplos_por_categoria(categoria: str) -> pd.DataFrame:
    """Devuelve un DataFrame de hasta 25 filas con productos de ejemplo
    para una categoria: [Producto, Verificado].
    """
    gold, catalogo = _cargar_fuentes()

    desde_gold = (
        gold.loc[gold["categoria_gold"] == categoria, "description_clean"]
        .drop_duplicates()
        .head(N_POR_CATEGORIA)
        .tolist()
    )

    filas = [(prod, "✓") for prod in desde_gold]

    faltan = N_POR_CATEGORIA - len(filas)
    if faltan > 0:
        ya_usados = set(desde_gold)
        candidatos = catalogo[
            (catalogo["categoria_final"] == categoria)
            & (catalogo["categoria_regla"] == catalogo["categoria_predicha"])
            & (~catalogo["description_clean"].isin(ya_usados))
        ].sort_values("revenue_total", ascending=False)

        for prod in candidatos["description_clean"].drop_duplicates().tolist():
            if faltan <= 0:
                break
            filas.append((prod, ""))
            faltan -= 1

    return pd.DataFrame(filas, columns=[COL_PRODUCTO, COL_VERIFICADO])


def opciones_dropdown() -> list[str]:
    """Etiquetas 'Categoría (English gloss)' para el selector de la UI,
    en el mismo orden de prioridad de las reglas.
    """
    return [f"{cat} ({GLOSA_EN[cat]})" for cat in CATEGORIAS_TABLA]


def categoria_desde_opcion(opcion: str) -> str:
    """Inversa de opciones_dropdown: recupera el nombre de categoria en
    español a partir de la etiqueta bilingue mostrada en el dropdown.
    """
    return opcion.rsplit(" (", 1)[0]
