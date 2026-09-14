"""
Clasificador de productos: normalizacion + reglas de keywords + modelo ML
(TF-IDF + LinearSVC) con fallback a reglas por umbral de confianza.

Este modulo porta, sin modificar la logica, el pipeline desarrollado en
product_classifier.ipynb (secciones 5.1, 5.9, 6 y 7). El notebook original
no expone ninguna funcion reutilizable: todo vive inline sobre columnas de
DataFrame. Este archivo es la version "empaquetada" de ese pipeline.

No se versiona ningun modelo serializado (.pkl / .joblib): el entrenamiento
tarda menos de un segundo sobre los ~2.880 ejemplos de train, asi que la app
entrena una sola vez al arrancar el proceso (ver `obtener_modelo`, cacheada).
Esto evita el riesgo de que un pickle generado con la version de sklearn del
entorno de desarrollo no cargue en el runtime del Hugging Face Space.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = Path(__file__).parent / "data"
RUTA_CATALOGO = DATA_DIR / "products_categorized.csv"
RUTA_GOLD = DATA_DIR / "gold_labels.csv"

# Mismo umbral que el notebook (celda 107, seccion 7): por debajo de este
# margen de confianza del modelo, se prefiere la etiqueta de la regla de
# keywords (o "Sin clasificar" si tampoco hay regla que matchee).
UMBRAL_CONFIANZA = 0.2

# -----------------------------------------------------------------------
# Reglas de keywords (product_classifier.ipynb, celda 59, seccion 5.1)
# -----------------------------------------------------------------------
# El orden del dict ES el orden de prioridad: gana la primera categoria
# cuya alternacion de regex matchea el texto. Copiado tal cual del
# notebook, incluidos los comentarios de fix (commits 439970b y fa92878).
REGLAS_CATEGORIAS = {
    # Estas dos categorias se agregaron para vaciar "Otros", que era la unica
    # etiqueta del gold que el modelo no podia predecir: entrena con las
    # etiquetas de las reglas, y las reglas nunca devuelven "Otros".
    # Van PRIMERO a proposito, porque el orden del dict es el orden de
    # prioridad: "MINI PLAYING CARDS SPACEBOY" es un juego antes que una
    # tarjeta, y "SET 12 COLOURING PENCILS DOILY" es escritura antes que
    # papeleria.
    "Juguetes y Juegos": [
        r"jigsaws?", r"puzzles?", r"playing cards", r"\bgames?\b", r"skittles",
        r"spinning tops?", r"yo.?yo", r"snakes? and ladders", r"dominoes",
        r"marbles", r"rocking horses?", r"\bdolls?\b", r"teddy|teddies",
        r"soldiers?", r"\btoys?\b", r"water bombs?", r"\bkites?\b",
        r"jack in the box",
    ],
    "Escritura y Útiles": [
        r"pencils?", r"\bpens?\b", r"sharpeners?", r"sharpner", r"crayons?",
        r"erasers?", r"\brubbers?\b", r"rulers?", r"\bchalk\b", r"colouring",
        r"\bpaints?\b",
    ],
    "Navidad": [
        r"christmas", r"xmas", r"santa", r"snowman", r"reindeer",
        r"holly", r"robin", r"stockings?", r"baubles?", r"advent",
        r"mistletoe", r"nativity",
    ],
    "Pascua y Fiestas": [
        # FIX: se saco r"birthdays?". Es un motivo impreso, no un tipo de
        # producto, y le robaba las tarjetas de cumpleanios a Papeleria y
        # Tarjeteria. Las guirnaldas de cumpleanios las sigue tomando buntings?.
        r"easter", r"bunn(y|ies)", r"buntings?",
        r"bal+oons?", r"party invites?", r"party bags?", r"invitations?",
    ],
    "Botellas de Agua Caliente y Confort": [
        r"hot water bottles?", r"hand warmers?", r"foot warmers?",
        r"hotties?",
    ],
    "Joyería y Bijouterie": [
        r"earrings?", r"bracelets?", r"necklaces?", r"pendants?",
        r"brooches?", r"bag charms?", r"phone charms?", r"key ?rings?",
        r"hoops?", r"jewell?ery",
    ],
    "Bolsos y Carteras": [
        r"\bbags?\b", r"handbags?", r"shoppers?", r"rucksacks?",
        r"backpacks?", r"purses?", r"totes?", r"wallets?",
    ],
    "Marcos y Fotografía": [
        r"photo frames?", r"picture frames?", r"photo albums?",
        r"photo shelf", r"\bframes?\b",
    ],
    "Espejos, Relojes y Arte de Pared": [
        r"mirrors?", r"\bclocks?\b", r"wall art", r"canvas",
    ],
    "Jardín y Exterior": [
        r"gardens?", r"birdhouses?", r"birdcages?", r"plant pots?",
        r"parasols?", r"watering cans?", r"trowels?", r"\brakes?\b",
        r"bird feeders?", r"secateurs",
        # FIX: antes caia en Textiles del Hogar por \bmats?\b. Un felpudo es un
        # articulo de entrada/exterior, no un textil decorativo. Jardin tiene
        # mayor prioridad que Textiles, asi que agregarlo aca alcanza.
        r"door ?mats?",
    ],
    "Cajas y Almacenamiento": [
        r"trinket box(es)?", r"money box(es)?", r"sewing box(es)?",
        r"gift box(es)?", r"jewellery box(es)?", r"keepsake box(es)?",
        r"storage box(es)?", r"treasure box(es)?", r"memory box(es)?",
        r"book box(es)?", r"lunch box(es)?", r"snack box(es)?",
        r"nesting box(es)?", r"trinket pots?", r"drawer knobs?",
        r"storage", r"cabinets?", r"chest of drawers", r"\bchests?\b",
        r"coat racks?", r"coat hangers?",
    ],
    "Cocina y Mesa": [
        r"\bmugs?\b", r"\bbowls?\b", r"\bplates?\b", r"\bcups?\b",
        r"teapots?", r"kettles?", r"cutlery", r"\bspoons?\b",
        r"\bforks?\b", r"\bknife\b", r"\bknives\b", r"cake ?stands?",
        r"cake ?tins?", r"cake ?cases?", r"tea ?sets?", r"tea towels?",
        r"teacups?", r"tea glass(es)?", r"saucers?", r"aprons?",
        r"coasters?", r"placemats?", r"napkins?", r"chopsticks?",
        r"popcorn", r"baking", r"\bjam\b", r"kitchen scales?",
        r"\bpantry\b", r"bread bins?", r"biscuit tins?",
        r"tea coffee sugar", r"\bjugs?\b",
    ],
    "Textiles del Hogar": [
        r"cushions?", r"curtains?", r"\brugs?\b", r"bedspreads?",
        r"quilts?", r"blankets?", r"\bmats?\b",   # FIX: doormats? -> Jardin
    ],
    "Papelería y Tarjetería": [
        # pencils/pens/colouring/paints migraron a "Escritura y Útiles"
        r"greeting cards?", r"\bcards?\b", r"notebooks?", r"journals?",
        r"gift wraps?", r"\bwraps?\b", r"stickers?", r"stationery",
        r"envelopes?", r"paper craft", r"paper chain", r"ribbons?",
        r"gift tags?", r"felt ?craft", r"doil(y|ies)",
    ],
    "Carteles y Señalética": [
        r"metal signs?", r"\bsigns?\b", r"plaques?",
    ],
    "Velas e Iluminación": [
        r"candles?", r"t ?lights?", r"votives?", r"lanterns?",
        r"\blights?\b",
    ],
    "Decoración del Hogar": [
        r"decorations?", r"ornaments?", r"garlands?", r"wreaths?",
        r"wicker", r"hanging hearts?", r"\bmobiles?\b", r"chalkboards?",
        r"black ?boards?", r"memo ?boards?", r"building blocks?",
        r"alphabet blocks?", r"\bfans?\b",
    ],
}

# Prioridad: primera categoria que matchea.
ORDEN_PRIORIDAD = list(REGLAS_CATEGORIAS.keys())


def normalizar(texto: str) -> str:
    """Misma normalizacion que product_classifier.ipynb, celda 26:
    mayusculas, todo lo no alfanumerico -> espacio, strip.
    """
    return re.sub(r"[^A-Z0-9]+", " ", texto.upper()).strip()


def etiquetar_por_reglas(texto: str, reglas=REGLAS_CATEGORIAS, orden=ORDEN_PRIORIDAD) -> str | None:
    """product_classifier.ipynb, celda 61."""
    texto = texto.lower()
    for categoria in orden:
        patron = "|".join(reglas[categoria])
        if re.search(patron, texto):
            return categoria
    return None


def categorias_que_matchean(texto: str, reglas=REGLAS_CATEGORIAS) -> list[str]:
    """product_classifier.ipynb, celda 65."""
    texto = texto.lower()
    return [cat for cat, patrones in reglas.items() if re.search("|".join(patrones), texto)]


def _cargar_datos_entrenamiento() -> tuple[pd.Series, pd.Series]:
    """Reconstruye el X_train / y_train de la celda 86: filas del catalogo
    con categoria_regla asignada, excluyendo los stock_code del gold.
    """
    catalogo = pd.read_csv(RUTA_CATALOGO, encoding="utf-8")
    gold = pd.read_csv(RUTA_GOLD, encoding="utf-8")

    # El gold paso por Excel y puede traer espacios de relleno en
    # stock_code; sin este strip en ambos lados el anti-join no encuentra
    # coincidencias y el gold "se filtra" al train (ver celda 79 del
    # notebook, que documenta el mismo problema).
    catalogo["stock_code"] = catalogo["stock_code"].astype(str).str.strip()
    codigos_gold = set(gold["stock_code"].astype(str).str.strip())

    train_data = catalogo[
        catalogo["categoria_regla"].notna() & ~catalogo["stock_code"].isin(codigos_gold)
    ].copy()

    return train_data["description_clean"], train_data["categoria_regla"]


def _entrenar_pipeline(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """product_classifier.ipynb, celdas 88 + 93, con el mejor C ya fijado
    (C=5, hallado por el GridSearchCV del notebook) para no repetir la
    busqueda de hiperparametros en cada arranque de la app.
    """
    vectorizador = FeatureUnion([
        ("palabras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
        ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ])
    pipeline = Pipeline([
        ("tfidf", vectorizador),
        ("clf", LinearSVC(C=5, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


@lru_cache(maxsize=1)
def obtener_modelo() -> Pipeline:
    """Entrena el pipeline una sola vez por proceso (~1 segundo) y lo
    cachea. No hay .joblib versionado: ver docstring del modulo.
    """
    X_train, y_train = _cargar_datos_entrenamiento()
    return _entrenar_pipeline(X_train, y_train)


def clasificar(texto: str) -> dict:
    """Clasifica un texto de producto libre, replicando la logica de
    umbral de confianza de las celdas 100 y 107 del notebook:

      1. Se normaliza el texto igual que en el entrenamiento.
      2. El modelo predice y se calcula el margen de confianza (diferencia
         entre el score top-1 y el top-2 de decision_function; LinearSVC no
         tiene predict_proba, asi que esto NO es una probabilidad).
      3. Si el margen >= UMBRAL_CONFIANZA, gana el modelo.
      4. Si no, gana la regla de keywords (o "Sin clasificar" si ninguna
         regla matchea tampoco).

    Devuelve un dict con la categoria final, quien decidio, el margen,
    el top-3 de categorias segun el modelo y lo que opinaba la regla.
    """
    texto_norm = normalizar(texto or "")
    if not texto_norm:
        return {
            "error": True,
            "mensaje": "Escribí el nombre de un producto (en inglés).",
        }

    modelo = obtener_modelo()
    clases = modelo.classes_

    scores = modelo.decision_function([texto_norm])[0]
    orden_desc = np.argsort(scores)[::-1]
    top3 = [(clases[i], float(scores[i])) for i in orden_desc[:3]]

    margen = float(scores[orden_desc[0]] - scores[orden_desc[1]])
    categoria_modelo = clases[orden_desc[0]]
    categoria_regla = etiquetar_por_reglas(texto_norm)

    if margen >= UMBRAL_CONFIANZA:
        categoria_final = categoria_modelo
        decidido_por = "modelo"
    else:
        categoria_final = categoria_regla or "Sin clasificar"
        decidido_por = "regla" if categoria_regla else "ninguno"

    return {
        "error": False,
        "texto_normalizado": texto_norm,
        "categoria": categoria_final,
        "decidido_por": decidido_por,
        "margen": margen,
        "top3": top3,
        "categoria_regla": categoria_regla,
    }
