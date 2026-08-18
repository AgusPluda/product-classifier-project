"""Verifica el .pbix regenerado contra el backup."""
import hashlib, json, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ok = True

def check(cond, msg):
    global ok
    print(("  OK   " if cond else "  FALLA") + " " + msg)
    if not cond:
        ok = False

src = zipfile.ZipFile(ROOT / "powerbi_dashboards.backup.pbix")
dst = zipfile.ZipFile(ROOT / "powerbi_dashboards.pbix")

print("== integridad del paquete ==")
check(dst.testzip() is None, "el ZIP abre sin errores")
check(
    hashlib.sha256(src.read("DataModel")).hexdigest()
    == hashlib.sha256(dst.read("DataModel")).hexdigest(),
    "DataModel identico byte a byte al backup",
)
check("SecurityBindings" not in dst.namelist(), "SecurityBindings eliminado")
check(b"SecurityBindings" not in dst.read("[Content_Types].xml"), "Override de SecurityBindings eliminado")
check(set(src.namelist()) - set(dst.namelist()) == {"SecurityBindings"}, "no falta ninguna otra parte")

lay = json.loads(dst.read("Report/Layout").decode("utf-16-le"))
print("\n== paginas ==")
check(len(lay["sections"]) == 4, "4 secciones")
names = [s["displayName"] for s in lay["sections"]]
check(names[3] == "Detalle de producto", "typo corregido: " + names[3])

group_ids = set()
for s in lay["sections"]:
    print("\n-- " + s["displayName"] + " --")
    for v in s["visualContainers"]:
        cfg = json.loads(v["config"])
        sv = cfg.get("singleVisual")
        if not sv:
            group_ids.add(cfg["name"])
            continue
        pos = cfg["layouts"][0]["position"]
        synced = all(abs(pos[k] - v[k]) < 0.001 for k in ("x", "y", "width", "height"))
        inside = v["x"] >= 0 and v["y"] >= 0 and v["x"] + v["width"] <= 1280.001 and v["y"] + v["height"] <= 720.001
        title = sv.get("vcObjects", {}).get("title", [{}])[0].get("properties", {}).get("text", {})
        title = title.get("expr", {}).get("Literal", {}).get("Value", "") if title else ""
        flag = "" if (synced and inside) else ("  <-- " + ("desincronizado " if not synced else "") + ("fuera del lienzo" if not inside else ""))
        print("    %-18s x%4d y%3d w%4d h%3d  %s%s" % (
            sv["visualType"], v["x"], v["y"], v["width"], v["height"], title, flag))
        check(synced, sv["visualType"] + ": posicion sincronizada")
        check(inside, sv["visualType"] + ": dentro del lienzo 1280x720")
        check("parentGroupName" not in cfg or cfg["parentGroupName"] in group_ids,
              sv["visualType"] + ": sin parentGroupName huerfano")

print("\n== correcciones puntuales ==")
resumen, estacionalidad, geografia, detalle = lay["sections"]

line = [v for v in resumen["visualContainers"] if json.loads(v["config"]).get("singleVisual", {}).get("visualType") == "lineChart"][0]
lc = json.loads(line["config"])["singleVisual"]
check("Series" not in lc["projections"], "Resumen/lineChart ya no duplica Estacionalidad (sin Series)")
lq = json.loads(line["query"])["Commands"][0]["SemanticQueryDataShapeCommand"]
sel_names = [s.get("Name") for s in lq["Query"]["Select"]]
check(not any("categoria_final" in (n or "") for n in sel_names), "Resumen/lineChart: Select sin categoria_final -> " + str(sel_names))
proj_idx = [i for g in lq["Binding"]["Primary"]["Groupings"] for i in g["Projections"]]
check(all(i < len(sel_names) for i in proj_idx), "Resumen/lineChart: Binding reindexado " + str(proj_idx))
check(not any("categoria_final" in json.dumps(s) for s in json.loads(line["dataTransforms"])["selects"]),
      "Resumen/lineChart: dataTransforms.selects limpio")

cards = [v for v in resumen["visualContainers"] if json.loads(v["config"]).get("singleVisual", {}).get("visualType") == "cardVisual"]
measures = [json.loads(v["config"])["singleVisual"]["projections"]["Data"][0]["queryRef"] for v in cards]
check(len(cards) == 4, "Resumen: 4 tarjetas KPI")
check("Medidas.Cantidad de Transacciones" in measures, "Resumen: KPI de transacciones -> " + str(measures))
check(all('"Property":"region"' not in (v.get("query") or "") for v in cards), "Resumen: KPIs sin el filtro region cacheado")

sl = [v for v in resumen["visualContainers"] if json.loads(v["config"]).get("singleVisual", {}).get("visualType") == "slicer"][0]
gen = json.loads(sl["config"])["singleVisual"]["objects"]["general"][0]["properties"]
check("filter" not in gen, "Resumen: slicer de region sin seleccion guardada (abre con todas)")

mapa = geografia["visualContainers"][4] if len(geografia["visualContainers"]) > 4 else None
mapa = [v for v in geografia["visualContainers"] if json.loads(v["config"]).get("singleVisual", {}).get("visualType") == "filledMap"][0]
mv = json.loads(mapa["config"])["singleVisual"]
check("Gradient" in mv["projections"] and "Tooltips" not in mv["projections"], "Geografia: mapa con Revenue en Gradient")
mdt = json.loads(mapa["dataTransforms"])
check(mdt["selects"][1]["roles"] == {"Gradient": True}, "Geografia: dataTransforms.selects con rol Gradient")
check(any(r["Name"] == "Gradient" for r in mdt["visualElements"][0]["DataRoles"]), "Geografia: DataRoles con Gradient")

tp = [v for v in resumen["visualContainers"] if json.loads(v["config"]).get("singleVisual", {}).get("visualType") == "clusteredBarChart"]
check(len(tp) == 2, "Resumen: barras de categoria + Top 10 productos")
top = [v for v in tp if "description_clean" in v["config"]][0]
check('"Top":10' in top["filters"], "Resumen: Top 10 aplicado al grafico de productos")
check("categoria_final" not in top["config"], "Resumen: Top 10 no arrastra categoria_final")

ids = [json.loads(v["config"])["name"] for s in lay["sections"] for v in s["visualContainers"]]
check(len(ids) == len(set(ids)), "todos los ids de visual son unicos")

theme = json.loads(dst.read("Report/StaticResources/RegisteredResources/Product_Classifier5570029722283252.json").decode("utf-8-sig"))
local = json.loads((ROOT / "powerbi_theme.json").read_text(encoding="utf-8"))
check(theme == local, "tema embebido == powerbi_theme.json")
check(theme["dataColors"][0] == "#2a78d6", "tema: acento primero en dataColors")

print("\n" + ("TODO OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)
