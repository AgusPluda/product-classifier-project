"""Rediseña el layout de powerbi_dashboards.pbix.

Lee SIEMPRE desde powerbi_dashboards.backup.pbix y escribe powerbi_dashboards.pbix,
así se puede re-ejecutar tantas veces como haga falta con resultado determinista.

Un .pbix es un ZIP; Report/Layout es JSON en UTF-16LE (sin BOM). Cada visualContainer
guarda su posición duplicada en x/y/z/width/height y en config.layouts[0].position:
hay que actualizar las dos.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "powerbi_dashboards.backup.pbix"
DST = ROOT / "powerbi_dashboards.pbix"
LAYOUT = "Report/Layout"
THEME_PART = "Report/StaticResources/RegisteredResources/Product_Classifier5570029722283252.json"
LOCAL_THEME = ROOT / "powerbi_theme.json"

# ---------------------------------------------------------------- sistema visual
INK = "#0b0b0b"
MUTED = "#52514e"
ACCENT = "#2a78d6"
HAIRLINE = "#e1e0d9"
SURFACE = "#ffffff"
CANVAS = "#fcfcfb"
CALLOUT_BG = "#eef4fd"

MARGIN = 24
HEADER_H = 72
BODY_TOP = 88
BODY_BOTTOM = 696


# ---------------------------------------------------------------- helpers de expr
def lit(value: str) -> dict:
    """Literal de Power BI: "'texto'", "12D", "true", "25L"."""
    return {"expr": {"Literal": {"Value": value}}}


def text_lit(value: str) -> dict:
    return lit("'" + value.replace("'", "''") + "'")


def solid(hex_color: str) -> dict:
    return {"solid": {"color": lit("'" + hex_color + "'")}}


def new_id() -> str:
    return secrets.token_hex(10)


# ------------------------------------------------------- helpers de contenedores
def place(container: dict, x: float, y: float, w: float, h: float, z: float) -> None:
    """Escribe la posición en los dos lugares donde Power BI la guarda."""
    container["x"], container["y"] = float(x), float(y)
    container["width"], container["height"] = float(w), float(h)
    container["z"] = float(z)
    layouts = container["_cfg"].setdefault("layouts", [{"id": 0, "position": {}}])
    pos = layouts[0].setdefault("position", {})
    pos.update({"x": float(x), "y": float(y), "z": float(z), "width": float(w), "height": float(h)})


def vco(container: dict) -> dict:
    return container["_cfg"]["singleVisual"].setdefault("vcObjects", {})


def set_title(container: dict, text: str, show: bool = True) -> None:
    props = {"show": lit("true" if show else "false")}
    if show:
        props.update(
            {
                "text": text_lit(text),
                "fontSize": lit("12D"),
                "bold": lit("true"),
                "fontColor": solid(INK),
                "alignment": lit("'left'"),
            }
        )
    vco(container)["title"] = [{"properties": props}]


def set_surface(container: dict, fill: str = SURFACE, radius: int = 6) -> None:
    """Tarjeta blanca con borde fino: el contenedor propio de cada visual."""
    vc = vco(container)
    vc["background"] = [
        {"properties": {"show": lit("true"), "color": solid(fill), "transparency": lit("0D")}}
    ]
    vc["border"] = [
        {"properties": {"show": lit("true"), "color": solid(HAIRLINE), "radius": lit(str(radius) + "D")}}
    ]


def set_chrome_off(container: dict) -> None:
    vc = vco(container)
    vc["title"] = [{"properties": {"show": lit("false")}}]
    vc["background"] = [{"properties": {"show": lit("false")}}]
    vc["border"] = [{"properties": {"show": lit("false")}}]


def set_obj(container: dict, obj: str, properties: dict) -> None:
    objects = container["_cfg"]["singleVisual"].setdefault("objects", {})
    entries = objects.setdefault(obj, [{"properties": {}}])
    entries[0].setdefault("properties", {}).update(properties)


# ------------------------------------------------------------ visuales sin query
def make_container(visual_type: str, x, y, w, h, z, objects: dict) -> dict:
    cfg = {
        "name": new_id(),
        "layouts": [
            {
                "id": 0,
                "position": {
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "width": float(w),
                    "height": float(h),
                },
            }
        ],
        "singleVisual": {
            "visualType": visual_type,
            "drillFilterOtherVisuals": True,
            "objects": objects,
            "vcObjects": {},
        },
    }
    container = {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "width": float(w),
        "height": float(h),
        "config": "",
        "filters": "[]",
        "_cfg": cfg,
    }
    set_chrome_off(container)
    return container


def rect(x, y, w, h, z, fill: str) -> dict:
    return make_container(
        "shape",
        x,
        y,
        w,
        h,
        z,
        {
            "shape": [{"properties": {"tileShape": lit("'rectangle'")}}],
            "fill": [
                {
                    "properties": {
                        "show": lit("true"),
                        "fillColor": solid(fill),
                        "transparency": lit("0D"),
                    }
                }
            ],
            "outline": [{"properties": {"show": lit("false")}}],
        },
    )


def textbox(
    x,
    y,
    w,
    h,
    z,
    text,
    size=11,
    color=MUTED,
    weight="400",
    align="left",
    family="Segoe UI",
    background=None,
):
    container = make_container(
        "textbox",
        x,
        y,
        w,
        h,
        z,
        {
            "general": [
                {
                    "properties": {
                        "paragraphs": [
                            {
                                "horizontalTextAlignment": align,
                                "textRuns": [
                                    {
                                        "value": text,
                                        "textStyle": {
                                            "fontFamily": family,
                                            "fontSize": str(size) + "pt",
                                            "fontWeight": weight,
                                            "color": color,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            ]
        },
    )
    if background:
        vco(container)["background"] = [
            {
                "properties": {
                    "show": lit("true"),
                    "color": solid(background),
                    "transparency": lit("0D"),
                }
            }
        ]
        vco(container)["border"] = [
            {"properties": {"show": lit("true"), "color": solid("#cde2fb"), "radius": lit("6D")}}
        ]
    return container


def page_header(section: dict, title: str, subtitle: str) -> None:
    """Banda de encabezado común a las 4 páginas."""
    band = rect(0, 0, 1280, HEADER_H, 0, SURFACE)
    hairline = rect(0, HEADER_H - 1, 1280, 1, 1, HAIRLINE)
    heading = textbox(
        MARGIN - 6, 10, 620, 30, 2, title, size=17, color=INK, weight="600", family="Segoe UI Semibold"
    )
    sub = textbox(MARGIN - 6, 41, 780, 22, 3, subtitle, size=10, color=MUTED)
    section["visualContainers"] = [band, hairline, heading, sub] + section["visualContainers"]


# --------------------------------------------------------------- clonado seguro
def clone(container: dict, replacements: dict) -> dict:
    """Clona un contenedor sustituyendo nombres de campo/medida en config, query y
    dataTransforms a la vez, y regenera el id para no duplicarlo."""
    new = {k: v for k, v in container.items() if k != "_cfg"}
    new = json.loads(json.dumps(new, ensure_ascii=False))
    for key in ("config", "query", "dataTransforms", "filters"):
        if isinstance(new.get(key), str) and new[key]:
            blob = new[key]
            for old, repl in replacements.items():
                blob = blob.replace(old, repl)
            new[key] = blob
    # separadores compactos: los patrones de reemplazo son sensibles a los espacios
    blob = json.dumps(container["_cfg"], ensure_ascii=False, separators=(",", ":"))
    for old, repl in replacements.items():
        blob = blob.replace(old, repl)
    new["_cfg"] = json.loads(blob)
    new["_cfg"]["name"] = new_id()
    return new


def _strip_where_from(data, prop: str) -> None:
    """Quita recursivamente las condiciones Where sobre una columna."""
    if isinstance(data, dict):
        where = data.get("Where")
        if isinstance(where, list):
            needle = '"Property":"' + prop + '"'
            kept = [
                w for w in where
                if needle not in json.dumps(w, separators=(",", ":"))
            ]
            if kept:
                data["Where"] = kept
            else:
                data.pop("Where")
        for value in list(data.values()):
            _strip_where_from(value, prop)
    elif isinstance(data, list):
        for value in data:
            _strip_where_from(value, prop)


def strip_where(container: dict, prop: str) -> None:
    """Limpia del query cacheado la condición de un slicer (Power BI la regenera al abrir)."""
    if container.get("query"):
        data = json.loads(container["query"])
        _strip_where_from(data, prop)
        container["query"] = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    _strip_where_from(container["_cfg"], prop)


def drop_role(container: dict, role: str) -> None:
    """Baja una proyección de un visual en todos los lugares donde vive."""
    sv = container["_cfg"]["singleVisual"]
    proj = sv.get("projections", {})
    if role not in proj:
        return
    refs = set(p["queryRef"] for p in proj.pop(role))

    dt = json.loads(container["dataTransforms"])
    removed = sorted(
        (i for i, sel in enumerate(dt["selects"]) if sel["queryName"] in refs), reverse=True
    )
    dt["selects"] = [s for s in dt["selects"] if s["queryName"] not in refs]
    dt.get("projectionOrdering", {}).pop(role, None)
    dt.get("projectionActiveItems", {}).pop(role, None)
    dt["queryMetadata"]["Select"] = [
        s for s in dt["queryMetadata"]["Select"] if s["Name"] not in refs
    ]

    def reindex(old):
        return old - sum(1 for i in removed if i < old)

    for element in dt.get("visualElements", []):
        element["DataRoles"] = [r for r in element["DataRoles"] if r["Name"] != role]
        for r in element["DataRoles"]:
            r["Projection"] = reindex(r["Projection"])
    for key, value in list(dt.get("projectionOrdering", {}).items()):
        dt["projectionOrdering"][key] = [reindex(i) for i in value if i not in removed]
    container["dataTransforms"] = json.dumps(dt, ensure_ascii=False, separators=(",", ":"))

    # prototypeQuery: solo hay que sacar el Select
    proto = sv["prototypeQuery"]
    proto["Select"] = [s for s in proto["Select"] if s.get("Name") not in refs]

    # query: sacar el Select y reindexar las Projections del Binding
    data = json.loads(container["query"])
    for command in data["Commands"]:
        cmd = command["SemanticQueryDataShapeCommand"]
        q = cmd["Query"]
        keep = [i for i, s in enumerate(q["Select"]) if s.get("Name") not in refs]
        q["Select"] = [q["Select"][i] for i in keep]
        binding = cmd.get("Binding", {})
        remap = {old: new for new, old in enumerate(keep)}
        for grouping in binding.get("Primary", {}).get("Groupings", []):
            for key in ("Projections", "SuppressedProjections"):
                if key in grouping:
                    grouping[key] = [remap[i] for i in grouping[key] if i in remap]
        if "SuppressedJoinPredicates" in binding:
            binding["SuppressedJoinPredicates"] = [
                remap[i] for i in binding["SuppressedJoinPredicates"] if i in remap
            ]
        if binding.pop("Secondary", None) is not None:
            # sin segunda agrupacion, el DataReduction de interseccion ya no aplica
            binding["DataReduction"] = {"DataVolume": 4, "Primary": {"BinnedLineSample": {}}}
    container["query"] = json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def move_role(container: dict, src_role: str, dst_role: str) -> None:
    """Mueve una proyección de un rol a otro (Tooltips -> Gradient en el mapa)."""
    sv = container["_cfg"]["singleVisual"]
    proj = sv["projections"]
    moved = proj.pop(src_role)
    proj[dst_role] = moved
    refs = set(p["queryRef"] for p in moved)

    dt = json.loads(container["dataTransforms"])
    ordering = dt.get("projectionOrdering", {})
    if src_role in ordering:
        ordering[dst_role] = ordering.pop(src_role)
    for element in dt.get("visualElements", []):
        for role in element["DataRoles"]:
            if role["Name"] == src_role:
                role["Name"] = dst_role
    for sel in dt["selects"]:
        if sel["queryName"] in refs and src_role in sel.get("roles", {}):
            sel["roles"].pop(src_role)
            sel["roles"][dst_role] = True
    container["dataTransforms"] = json.dumps(dt, ensure_ascii=False, separators=(",", ":"))


def clear_slicer_selection(container: dict) -> None:
    objects = container["_cfg"]["singleVisual"].setdefault("objects", {})
    for entry in objects.get("general", []):
        entry.get("properties", {}).pop("filter", None)


# ------------------------------------------------------------------- páginas
def restyle(layout: dict) -> None:
    for section in layout["sections"]:
        for c in section["visualContainers"]:
            c["_cfg"] = json.loads(c["config"])
        cfg = json.loads(section.get("config") or "{}")
        cfg.setdefault("objects", {})["background"] = [
            {"properties": {"color": solid(CANVAS), "transparency": lit("0D")}}
        ]
        section["config"] = json.dumps(cfg, ensure_ascii=False, separators=(",", ":"))

    resumen, estacionalidad, geografia, detalle = layout["sections"]

    # ---------------------------------------------------------- 1. Resumen
    _group, card_rev, card_uni, card_tkt, bars, slicer_region, line = resumen["visualContainers"]

    # disolver "Grupo 1": el contenedor del grupo se descarta y las tarjetas pasan a
    # coordenadas absolutas
    for card in (card_rev, card_uni, card_tkt):
        card["_cfg"].pop("parentGroupName", None)

    card_txn = clone(card_rev, {"Revenue Total": "Cantidad de Transacciones"})
    cards = [card_rev, card_uni, card_tkt, card_txn]
    kpi_w, kpi_h = 296, 104
    for i, card in enumerate(cards):
        place(card, MARGIN + i * (kpi_w + 16), BODY_TOP, kpi_w, kpi_h, 100 + i)
        set_surface(card)
        set_title(card, "", show=False)
        strip_where(card, "region")

    # el slicer de región abría el reporte filtrado a Reino Unido sin avisar
    clear_slicer_selection(slicer_region)
    strip_where(slicer_region, "region")
    place(slicer_region, 964, 16, 292, 40, 110)
    set_obj(slicer_region, "general", {"orientation": lit("2D")})
    set_obj(slicer_region, "header", {"show": lit("false")})
    vco(slicer_region)["title"] = [{"properties": {"show": lit("false")}}]

    strip_where(bars, "region")
    place(bars, MARGIN, 208, 760, BODY_BOTTOM - 208, 120)
    set_surface(bars)
    set_title(bars, "Revenue por categoría")
    set_obj(bars, "dataPoint", {"defaultColor": solid(ACCENT), "fillTransparency": lit("0D")})
    set_obj(bars, "legend", {"show": lit("false")})

    # el lineChart de Resumen era un duplicado exacto del de Estacionalidad:
    # sin la serie por categoría queda la tendencia total, que sí es de resumen
    drop_role(line, "Series")
    strip_where(line, "region")
    place(line, 808, 208, 448, 236, 130)
    set_surface(line)
    set_title(line, "Tendencia mensual (meses completos)")
    set_obj(line, "legend", {"show": lit("false")})
    set_obj(line, "lineStyles", {"showMarker": lit("true"), "strokeWidth": lit("2D")})

    top_products = clone(
        bars,
        {
            "dim_producto.categoria_final": "dim_producto.description_clean",
            '"Property":"categoria_final"': '"Property":"description_clean"',
            '"NativeReferenceName":"categoria_final"': '"NativeReferenceName":"description_clean"',
            '"Restatement":"categoria_final"': '"Restatement":"description_clean"',
            '"displayName":"categoria_final"': '"displayName":"description_clean"',
        },
    )
    # reutiliza el filtro Top N que ya existía en la tabla de la página 4
    top_products["filters"] = detalle["visualContainers"][0]["filters"].replace('"Top":15', '"Top":10')
    place(top_products, 808, 460, 448, BODY_BOTTOM - 460, 140)
    set_surface(top_products)
    set_title(top_products, "Top 10 productos por revenue")
    set_obj(top_products, "categoryAxis", {"fontSize": lit("9D"), "maxMarginFactor": lit("40L")})
    set_obj(top_products, "labels", {"show": lit("false")})

    resumen["visualContainers"] = [
        textbox(880, 26, 76, 22, 111, "Región", size=10, color=MUTED, align="right"),
        slicer_region,
        bars,
        line,
        top_products,
    ] + cards
    page_header(
        resumen,
        "Resumen ejecutivo",
        "Online Retail II · dic 2009 – dic 2011 · 1.003.417 líneas de transacción · 4.724 productos clasificados",
    )

    # --------------------------------------------------- 2. Estacionalidad
    line2, slicer_cat = estacionalidad["visualContainers"]
    place(line2, MARGIN, BODY_TOP, 984, BODY_BOTTOM - BODY_TOP, 100)
    set_surface(line2)
    set_title(line2, "Revenue mensual por categoría")
    set_obj(line2, "lineStyles", {"showMarker": lit("true"), "strokeWidth": lit("2D")})
    set_obj(line2, "legend", {"show": lit("true"), "position": lit("'Top'"), "showTitle": lit("false")})

    place(slicer_cat, 1032, BODY_TOP, 224, BODY_BOTTOM - BODY_TOP, 110)
    set_surface(slicer_cat)
    set_obj(slicer_cat, "header", {"show": lit("true"), "fontSize": lit("11D")})
    vco(slicer_cat)["title"] = [{"properties": {"show": lit("false")}}]
    page_header(
        estacionalidad,
        "Estacionalidad por categoría",
        "Diciembre 2011 excluido: es un mes incompleto (los datos cortan el 9/12/2011) y distorsiona la comparación mes a mes",
    )

    # -------------------------------------------------------- 3. Geografía
    mapa, tabla_geo = geografia["visualContainers"]
    # Revenue Total estaba en Tooltips: el mapa se veía todo del mismo tono
    move_role(mapa, "Tooltips", "Gradient")
    place(mapa, MARGIN, BODY_TOP, 712, BODY_BOTTOM - BODY_TOP, 100)
    set_surface(mapa)
    set_title(mapa, "Revenue por país")

    callout = textbox(
        760,
        BODY_TOP,
        496,
        88,
        110,
        "Reino Unido concentra ~92% del revenue total. Los demás países se reparten el 8% "
        "restante: las diferencias entre ellos son de una escala muy distinta.",
        size=10,
        color="#1b3a63",
        background=CALLOUT_BG,
    )
    place(tabla_geo, 760, 200, 496, BODY_BOTTOM - 200, 120)
    set_surface(tabla_geo)
    set_title(tabla_geo, "Mix de categorías por región")
    geografia["visualContainers"] = [mapa, callout, tabla_geo]
    page_header(
        geografia,
        "Geografía",
        "Revenue por país y comparación del mix de categorías entre Reino Unido y el resto del mundo",
    )

    # ------------------------------------------------ 4. Detalle de producto
    detalle["displayName"] = "Detalle de producto"
    tabla_prod, slicer_prod = detalle["visualContainers"]
    place(slicer_prod, MARGIN, BODY_TOP, 248, BODY_BOTTOM - BODY_TOP, 100)
    set_surface(slicer_prod)
    set_obj(slicer_prod, "header", {"show": lit("true"), "fontSize": lit("11D")})
    vco(slicer_prod)["title"] = [{"properties": {"show": lit("false")}}]

    place(tabla_prod, 296, BODY_TOP, 960, BODY_BOTTOM - BODY_TOP, 110)
    set_surface(tabla_prod)
    set_title(tabla_prod, "Productos por revenue (Top 15 de la selección)")
    detalle["visualContainers"] = [slicer_prod, tabla_prod]
    page_header(
        detalle,
        "Detalle de producto",
        "Top 15 productos por revenue dentro de la categoría seleccionada",
    )

    # ------------------------------------------------------------ serializar
    for section in layout["sections"]:
        for c in section["visualContainers"]:
            c["config"] = json.dumps(c.pop("_cfg"), ensure_ascii=False, separators=(",", ":"))
            c.setdefault("filters", "[]")


# --------------------------------------------------------------------- tema
def restyle_theme(theme: dict) -> dict:
    # Power BI anexa ~200 colores heredados al guardar; nos quedamos con la paleta
    # curada (acento primero) y dejamos que el tema base cubra el resto.
    palette = [ACCENT] + [c for c in theme.get("dataColors", []) if c != ACCENT]
    theme["dataColors"] = palette[:8]
    theme.update(
        {
            "name": "Product Classifier",
            "background": CANVAS,
            "foreground": INK,
            "foregroundNeutralSecondary": MUTED,
            "tableAccent": ACCENT,
            "maximum": "#0d366b",
            "center": "#7fb0e8",
            "minimum": "#cde2fb",
        }
    )
    theme["textClasses"] = {
        "callout": {"fontSize": 28, "fontFace": "Segoe UI Semibold", "color": INK},
        "title": {"fontSize": 12, "fontFace": "Segoe UI Semibold", "color": INK},
        "header": {"fontSize": 11, "fontFace": "Segoe UI Semibold", "color": INK},
        "label": {"fontSize": 10, "fontFace": "Segoe UI", "color": MUTED},
    }
    grid = {
        "gridlineColor": {"solid": {"color": HAIRLINE}},
        "axisColor": "#c3c2b7",
        "fontSize": 10,
        "labelColor": {"solid": {"color": MUTED}},
    }
    theme["visualStyles"] = {
        "*": {
            "*": {
                # el título de cada visual se define en su vcObjects; acá solo la tipografía
                "title": [
                    {
                        "show": True,
                        "fontColor": {"solid": {"color": INK}},
                        "fontSize": 12,
                        "bold": True,
                        "alignment": "left",
                    }
                ],
                "background": [{"show": False}],
                "border": [{"show": False}],
                "dropShadow": [{"show": False}],
                "legend": [{"labelColor": {"solid": {"color": MUTED}}, "fontSize": 10}],
                "labels": [{"color": {"solid": {"color": MUTED}}, "fontSize": 9}],
            }
        },
        "page": {"*": {"background": [{"color": {"solid": {"color": CANVAS}}, "transparency": 0}]}},
        "barChart": {"*": {"categoryAxis": [grid], "valueAxis": [grid]}},
        "columnChart": {"*": {"categoryAxis": [grid], "valueAxis": [grid]}},
        "lineChart": {"*": {"categoryAxis": [grid], "valueAxis": [grid]}},
        "tableEx": {
            "*": {
                "grid": [
                    {"gridVertical": False, "gridHorizontalColor": {"solid": {"color": HAIRLINE}}}
                ],
                "columnHeaders": [
                    {"fontColor": {"solid": {"color": INK}}, "bold": True, "fontSize": 10}
                ],
                "values": [{"fontSize": 10, "fontColor": {"solid": {"color": INK}}}],
            }
        },
        "slicer": {"*": {"header": [{"fontColor": {"solid": {"color": MUTED}}, "fontSize": 11}]}},
    }
    return theme


# ------------------------------------------------------------------ empaquetado
def main() -> int:
    if not SRC.exists():
        print("falta el backup: " + str(SRC), file=sys.stderr)
        return 1

    with zipfile.ZipFile(SRC) as z:
        parts = {i.filename: (i, z.read(i.filename)) for i in z.infolist()}

    layout = json.loads(parts[LAYOUT][1].decode("utf-16-le"))
    restyle(layout)
    parts[LAYOUT] = (
        parts[LAYOUT][0],
        json.dumps(layout, ensure_ascii=False, separators=(",", ":")).encode("utf-16-le"),
    )

    theme = restyle_theme(json.loads(parts[THEME_PART][1].decode("utf-8-sig")))
    theme_text = json.dumps(theme, ensure_ascii=False, indent=2)
    parts[THEME_PART] = (parts[THEME_PART][0], theme_text.encode("utf-8"))
    LOCAL_THEME.write_text(theme_text + "\n", encoding="utf-8")

    # SecurityBindings firma el contenido: si se deja tras editar Report/Layout,
    # Power BI Desktop rechaza el archivo.
    parts.pop("SecurityBindings", None)
    ct_name = "[Content_Types].xml"
    info, raw = parts[ct_name]
    parts[ct_name] = (info, re.sub(rb'<Override PartName="/SecurityBindings"[^>]*/>', b"", raw))

    tmp = DST.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w") as out:
        for name, (info, data) in parts.items():
            zi = zipfile.ZipInfo(name, date_time=info.date_time)
            zi.compress_type = info.compress_type
            zi.external_attr = info.external_attr
            zi.create_system = info.create_system
            out.writestr(zi, data)
    shutil.move(str(tmp), str(DST))
    print("escrito " + str(DST) + " (" + format(DST.stat().st_size, ",") + " bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
