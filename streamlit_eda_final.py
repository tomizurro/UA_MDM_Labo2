from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Configuración ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Final — PetFinder",
    page_icon="🐾",
    layout="wide",
)

DATA_PATH   = Path("input/petfinder-adoption-prediction/train/train.csv")
BREED_PATH  = Path("input/petfinder-adoption-prediction/breed_labels.csv")
COLOR_PATH  = Path("input/petfinder-adoption-prediction/color_labels.csv")
STATE_PATH  = Path("input/petfinder-adoption-prediction/state_labels.csv")

TARGET_COL  = "AdoptionSpeed"
NUMERIC_COLS = ["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]
CORR_COLS    = ["Age", "Fee", "Quantity", "VideoAmt", "PhotoAmt",
                "Vaccinated", "Dewormed", "Sterilized", "Health",
                "MaturitySize", "FurLength", "Gender", "AdoptionSpeed"]

TYPE_MAP    = {1: "Perro", 2: "Gato"}
GENDER_MAP  = {1: "Macho", 2: "Hembra", 3: "Mixto"}
SIZE_MAP    = {0: "No especificado", 1: "Pequeño", 2: "Mediano", 3: "Grande", 4: "Extra Grande"}
FUR_MAP     = {0: "No especificado", 1: "Corto", 2: "Medio", 3: "Largo"}
YES_NO_MAP  = {1: "Sí", 2: "No", 3: "No sabe"}
HEALTH_MAP  = {0: "No especificado", 1: "Sano", 2: "Lesión leve", 3: "Lesión grave"}
TARGET_LABELS = {
    0: "0 — Mismo día",
    1: "1 — 1 a 7 días",
    2: "2 — 8 a 30 días",
    3: "3 — 31 a 90 días",
    4: "4 — No adoptada",
}
AGE_BINS   = [-1, 6, 12, 24, 60, float("inf")]
AGE_LABELS = ["0–6 m", "7–12 m", "13–24 m", "25–60 m", "60+ m"]

PALETTE = px.colors.qualitative.Set2


# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    breed_labels = pd.read_csv(BREED_PATH)
    state_labels = pd.read_csv(STATE_PATH)

    breed_map = breed_labels.set_index("BreedID")["BreedName"].to_dict()
    state_map = state_labels.set_index("StateID")["StateName"].to_dict()

    df["Type_label"]         = df["Type"].map(TYPE_MAP)
    df["Gender_label"]       = df["Gender"].map(GENDER_MAP)
    df["MaturitySize_label"] = df["MaturitySize"].map(SIZE_MAP)
    df["FurLength_label"]    = df["FurLength"].map(FUR_MAP)
    df["Vaccinated_label"]   = df["Vaccinated"].map(YES_NO_MAP)
    df["Dewormed_label"]     = df["Dewormed"].map(YES_NO_MAP)
    df["Sterilized_label"]   = df["Sterilized"].map(YES_NO_MAP)
    df["Health_label"]       = df["Health"].map(HEALTH_MAP)
    df["AdoptionSpeedLabel"] = df[TARGET_COL].map(TARGET_LABELS)
    df["Breed1_label"]       = df["Breed1"].map(breed_map)
    df["State_label"]        = df["State"].map(state_map)
    df["AgeGroup"]           = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["Quantity_group"]     = df["Quantity"].apply(
        lambda x: "1" if x == 1 else ("2" if x == 2 else "3+")
    )
    df["has_fee"]            = (df["Fee"] > 0).map({True: "Con costo", False: "Gratuita"})
    df["desc_len"]           = df["Description"].fillna("").apply(len)
    df["desc_words"]         = df["Description"].fillna("").apply(lambda x: len(x.split()))
    return df


# ── Sidebar / Filtros ──────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros globales")

    tipo = st.sidebar.selectbox(
        "Tipo de mascota",
        ["Todos", "Perro", "Gato"],
    )
    speeds = st.sidebar.multiselect(
        "AdoptionSpeed",
        sorted(df[TARGET_COL].unique()),
        default=sorted(df[TARGET_COL].unique()),
        help="0 = adoptada el mismo día · 4 = no adoptada",
    )
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Edad (meses)", age_min, age_max, (age_min, age_max))

    filtered = df.copy()
    if tipo != "Todos":
        filtered = filtered[filtered["Type_label"] == tipo]
    filtered = filtered[filtered[TARGET_COL].isin(speeds)]
    filtered = filtered[filtered["Age"].between(*age_range)]
    return filtered


# ── Helpers ────────────────────────────────────────────────────────────────────
def proportion_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df.groupby(col)[TARGET_COL]
        .value_counts(normalize=True)
        .mul(100).round(2)
        .rename("pct")
        .reset_index()
    )


def corr_heatmap(df: pd.DataFrame, cols: list, title: str) -> go.Figure:
    corr = df[cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title,
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ── Secciones ──────────────────────────────────────────────────────────────────

def tab_resumen(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    st.subheader("Resumen del dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros totales", f"{len(df):,}")
    c2.metric("Registros filtrados", f"{len(filtered):,}")
    c3.metric("Perros", f"{(df['Type_label'] == 'Perro').sum():,}")
    c4.metric("Gatos", f"{(df['Type_label'] == 'Gato').sum():,}")

    st.divider()
    left, right = st.columns([1.4, 1])

    with left:
        st.caption("Primeros registros (filtrado activo)")
        st.dataframe(filtered.head(20), use_container_width=True)

    with right:
        st.caption("Diccionario de variables")
        dct = pd.DataFrame([
            ("Type",         "Tipo de mascota",                    "1=Perro, 2=Gato"),
            ("Name",         "Nombre",                             "Vacío si no tiene"),
            ("Age",          "Edad al publicarse",                 "En meses"),
            ("Breed1/2",     "Raza principal / secundaria",        "Ver breed_labels"),
            ("Gender",       "Género",                             "1=Macho, 2=Hembra, 3=Mixto"),
            ("Color1/2/3",   "Colores del animal",                 "Ver color_labels"),
            ("MaturitySize", "Tamaño adulto",                      "0–4"),
            ("FurLength",    "Largo del pelaje",                   "0–3"),
            ("Vaccinated",   "Vacunado",                           "1=Sí, 2=No, 3=NS"),
            ("Dewormed",     "Desparasitado",                      "1=Sí, 2=No, 3=NS"),
            ("Sterilized",   "Esterilizado",                       "1=Sí, 2=No, 3=NS"),
            ("Health",       "Condición de salud",                 "0–3"),
            ("Quantity",     "Mascotas en el perfil",              "Numérico"),
            ("Fee",          "Costo de adopción",                  "0 = gratuito"),
            ("State",        "Estado en Malasia",                  "Ver state_labels"),
            ("PhotoAmt",     "Cantidad de fotos",                  "Numérico"),
            ("VideoAmt",     "Cantidad de videos",                 "Numérico"),
            ("Description",  "Texto descriptivo del perfil",       "Texto libre"),
            ("AdoptionSpeed","Velocidad de adopción (TARGET)",      "0=mismo día … 4=no adoptada"),
        ], columns=["Variable", "Descripción", "Valores"])
        st.dataframe(dct, use_container_width=True, hide_index=True)


def tab_calidad(df: pd.DataFrame) -> None:
    st.subheader("Calidad de datos")

    summary = pd.DataFrame({
        "dtype":       df.dtypes.astype(str),
        "n_unique":    df.nunique(),
        "missing":     df.isna().sum(),
        "missing_pct": df.isna().mean().mul(100).round(2),
    }).sort_values("missing", ascending=False)

    missing = summary[summary["missing"] > 0].reset_index().rename(columns={"index": "variable"})

    left, right = st.columns(2)
    with left:
        st.caption("Resumen de columnas")
        st.dataframe(summary, use_container_width=True)
    with right:
        if missing.empty:
            st.success("No hay valores faltantes.")
        else:
            fig = px.bar(
                missing, x="missing_pct", y="variable", orientation="h",
                title="% de valores faltantes por columna",
                color="missing_pct", color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


def tab_target(filtered: pd.DataFrame) -> None:
    st.subheader("Variable objetivo: AdoptionSpeed")

    tdf = (
        filtered[TARGET_COL].value_counts().sort_index()
        .rename_axis(TARGET_COL).reset_index(name="count")
    )
    tdf["pct"] = (tdf["count"] / len(filtered) * 100).round(2)
    tdf["label"] = tdf[TARGET_COL].map(TARGET_LABELS)

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            tdf, x=TARGET_COL, y="count", text="count",
            color=TARGET_COL, color_discrete_sequence=PALETTE,
            title="Distribución de AdoptionSpeed",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(
            tdf, values="count", names="label",
            title="Participación por clase", hole=0.45,
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(tdf[["label", "count", "pct"]], use_container_width=True, hide_index=True)


def tab_numericas(filtered: pd.DataFrame) -> None:
    st.subheader("Variables numéricas")
    st.dataframe(filtered[NUMERIC_COLS].describe().T.round(2), use_container_width=True)

    st.divider()
    var = st.selectbox("Variable", NUMERIC_COLS)
    left, right = st.columns(2)

    with left:
        fig = px.histogram(filtered, x=var, nbins=50, title=f"Distribución de {var}",
                           color_discrete_sequence=["#0f766e"])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.box(filtered, x=TARGET_COL, y=var, color=TARGET_COL,
                     color_discrete_sequence=PALETTE,
                     title=f"{var} vs AdoptionSpeed")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Análisis detallado de Quantity ──────────────────────────────────────
    st.divider()
    st.markdown("#### Análisis detallado de Quantity")

    qty_left, qty_right = st.columns(2)

    with qty_left:
        qty_grp = (
            filtered.groupby(["Quantity_group", TARGET_COL])
            .size().reset_index(name="count")
        )
        fig = px.bar(
            qty_grp, x="Quantity_group", y="count", color=TARGET_COL,
            barmode="group", title="AdoptionSpeed por agrupación de Quantity (1 / 2 / 3+)",
            category_orders={"Quantity_group": ["1", "2", "3+"]},
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with qty_right:
        qty_avg = (
            filtered.groupby("Quantity")
            .agg(mean_speed=(TARGET_COL, "mean"), count=("Quantity", "count"))
            .reset_index()
        )
        qty_avg = qty_avg[qty_avg["count"] >= 5]
        fig = px.scatter(
            qty_avg, x="Quantity", y="mean_speed", size="count",
            title="AdoptionSpeed promedio vs Quantity\n(tamaño = frecuencia)",
            labels={"mean_speed": "AdoptionSpeed promedio"},
            color_discrete_sequence=["#ef4444"],
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_categoricas(filtered: pd.DataFrame) -> None:
    st.subheader("Variables categóricas")

    col_map = {
        "Type_label":         "Tipo",
        "Gender_label":       "Género",
        "MaturitySize_label": "Tamaño adulto",
        "FurLength_label":    "Largo de pelaje",
        "Vaccinated_label":   "Vacunación",
        "Dewormed_label":     "Desparasitación",
        "Sterilized_label":   "Esterilización",
        "Health_label":       "Condición de salud",
        "has_fee":            "Fee (Gratuita vs Con costo)",
    }
    sel = st.selectbox("Variable categórica", list(col_map.keys()), format_func=col_map.get)

    left, right = st.columns(2)

    with left:
        cnt = filtered[sel].value_counts().reset_index()
        cnt.columns = [sel, "count"]
        fig = px.bar(cnt, x=sel, y="count", text="count",
                     title=f"Distribución — {col_map[sel]}",
                     color="count", color_continuous_scale="Purples")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        pct = proportion_df(filtered, sel)
        fig = px.bar(pct, x=sel, y="pct", color=TARGET_COL,
                     barmode="group",
                     title=f"{col_map[sel]} vs AdoptionSpeed (%)",
                     color_discrete_sequence=PALETTE,
                     labels={"pct": "% dentro de la categoría"})
        st.plotly_chart(fig, use_container_width=True)

    # Razas top
    st.divider()
    st.markdown("#### Top razas (Breed1)")
    top_n = st.slider("Cantidad de razas a mostrar", 5, 30, 15)
    top_breeds = filtered["Breed1_label"].value_counts().head(top_n).reset_index()
    top_breeds.columns = ["Raza", "count"]
    fig = px.bar(top_breeds.sort_values("count"), x="count", y="Raza",
                 orientation="h", title=f"Top {top_n} razas principales",
                 color="count", color_continuous_scale="Blues")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Estado geográfico
    st.divider()
    st.markdown("#### AdoptionSpeed promedio por Estado")
    state_avg = (
        filtered.groupby("State_label")[TARGET_COL]
        .mean().round(2).reset_index()
        .sort_values(TARGET_COL)
    )
    fig = px.bar(
        state_avg, x=TARGET_COL, y="State_label", orientation="h",
        title="AdoptionSpeed promedio por Estado (Malasia)",
        color=TARGET_COL, color_continuous_scale="RdYlGn_r",
        labels={TARGET_COL: "AdoptionSpeed promedio", "State_label": ""},
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def tab_bivariado(filtered: pd.DataFrame) -> None:
    st.subheader("Análisis bivariado: Perros vs Gatos respecto al target")
    st.caption("Cada gráfico compara Perros y Gatos en relación a AdoptionSpeed para distintas variables.")

    dogs = filtered[filtered["Type_label"] == "Perro"]
    cats = filtered[filtered["Type_label"] == "Gato"]

    # ── Género ────────────────────────────────────────────────────────────────
    st.markdown("#### Género")
    left, right = st.columns(2)
    gender_order = ["Macho", "Hembra", "Mixto"]
    for col, subset, label in [(left, dogs, "Perros"), (right, cats, "Gatos")]:
        with col:
            grp = subset.groupby(["Gender_label", TARGET_COL]).size().reset_index(name="count")
            fig = px.bar(grp, x="Gender_label", y="count", color=TARGET_COL,
                         barmode="group", title=f"Género vs AdoptionSpeed — {label}",
                         category_orders={"Gender_label": gender_order},
                         color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)

    tbl_gender = (
        filtered.groupby(["Type_label", "Gender_label"])[TARGET_COL]
        .mean().round(2).unstack()
    )
    st.caption("AdoptionSpeed promedio por género y tipo")
    st.dataframe(tbl_gender, use_container_width=True)

    st.divider()

    # ── Edad ─────────────────────────────────────────────────────────────────
    st.markdown("#### Edad")
    left, right = st.columns(2)
    for col, subset, label in [(left, dogs, "Perros"), (right, cats, "Gatos")]:
        with col:
            fig = px.box(subset, x=TARGET_COL, y="Age", color=TARGET_COL,
                         title=f"Edad vs AdoptionSpeed — {label}",
                         color_discrete_sequence=PALETTE,
                         labels={"Age": "Edad (meses)"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    tbl_age = (
        filtered.groupby(["Type_label", TARGET_COL])["Age"]
        .median().round(1).unstack()
    )
    st.caption("Mediana de edad por AdoptionSpeed y tipo")
    st.dataframe(tbl_age, use_container_width=True)

    st.divider()

    # ── Tamaño ────────────────────────────────────────────────────────────────
    st.markdown("#### Tamaño adulto")
    size_order = ["Pequeño", "Mediano", "Grande", "Extra Grande", "No especificado"]
    left, right = st.columns(2)
    for col, subset, label in [(left, dogs, "Perros"), (right, cats, "Gatos")]:
        with col:
            grp = subset.groupby(["MaturitySize_label", TARGET_COL]).size().reset_index(name="count")
            fig = px.bar(grp, x="MaturitySize_label", y="count", color=TARGET_COL,
                         barmode="group", title=f"Tamaño vs AdoptionSpeed — {label}",
                         category_orders={"MaturitySize_label": size_order},
                         color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)

    tbl_size = (
        filtered.groupby(["Type_label", "MaturitySize_label"])[TARGET_COL]
        .mean().round(2).unstack()
    )
    st.caption("AdoptionSpeed promedio por tamaño y tipo")
    st.dataframe(tbl_size, use_container_width=True)

    st.divider()

    # ── Vacunación ────────────────────────────────────────────────────────────
    st.markdown("#### Vacunación")
    vacc_order = ["Sí", "No", "No sabe"]
    left, right = st.columns(2)
    for col, subset, label in [(left, dogs, "Perros"), (right, cats, "Gatos")]:
        with col:
            grp = subset.groupby(["Vaccinated_label", TARGET_COL]).size().reset_index(name="count")
            fig = px.bar(grp, x="Vaccinated_label", y="count", color=TARGET_COL,
                         barmode="group", title=f"Vacunación vs AdoptionSpeed — {label}",
                         category_orders={"Vaccinated_label": vacc_order},
                         color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)

    tbl_vacc = (
        filtered.groupby(["Type_label", "Vaccinated_label"])[TARGET_COL]
        .mean().round(2).unstack()
    )
    st.caption("AdoptionSpeed promedio por vacunación y tipo")
    st.dataframe(tbl_vacc, use_container_width=True)

    st.divider()

    # ── Condición de salud ────────────────────────────────────────────────────
    st.markdown("#### Condición de salud")
    health_order = ["Sano", "Lesión leve", "Lesión grave", "No especificado"]
    left, right = st.columns(2)
    for col, subset, label in [(left, dogs, "Perros"), (right, cats, "Gatos")]:
        with col:
            grp = subset.groupby(["Health_label", TARGET_COL]).size().reset_index(name="count")
            fig = px.bar(grp, x="Health_label", y="count", color=TARGET_COL,
                         barmode="group", title=f"Salud vs AdoptionSpeed — {label}",
                         category_orders={"Health_label": health_order},
                         color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)

    tbl_health = (
        filtered.groupby(["Type_label", "Health_label"])[TARGET_COL]
        .mean().round(2).unstack()
    )
    st.caption("AdoptionSpeed promedio por condición de salud y tipo")
    st.dataframe(tbl_health, use_container_width=True)


def tab_correlacion(filtered: pd.DataFrame) -> None:
    st.subheader("Matrices de correlación")

    # General
    st.markdown("#### General")
    st.plotly_chart(
        corr_heatmap(filtered, CORR_COLS, "Correlación — todos"),
        use_container_width=True,
    )

    st.divider()

    # Por tipo
    st.markdown("#### Por tipo de mascota")
    left, right = st.columns(2)
    dogs = filtered[filtered["Type_label"] == "Perro"]
    cats = filtered[filtered["Type_label"] == "Gato"]

    with left:
        st.plotly_chart(
            corr_heatmap(dogs, CORR_COLS, "Correlación — Perros"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            corr_heatmap(cats, CORR_COLS, "Correlación — Gatos"),
            use_container_width=True,
        )

    # Outliers
    st.divider()
    st.markdown("#### Outliers en variables numéricas")
    out_var = st.selectbox("Variable", NUMERIC_COLS, key="outlier_var")
    fig = px.box(
        filtered, x="Type_label", y=out_var, color="Type_label",
        points="outliers", title=f"Outliers en {out_var} por tipo",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def tab_texto(filtered: pd.DataFrame) -> None:
    st.subheader("Análisis de texto: Description")

    left, right = st.columns(2)
    with left:
        fig = px.box(
            filtered, x=TARGET_COL, y="desc_len", color=TARGET_COL,
            color_discrete_sequence=PALETTE,
            title="Longitud de descripción (caracteres) vs AdoptionSpeed",
            labels={"desc_len": "Caracteres"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(
            filtered, x=TARGET_COL, y="desc_words", color=TARGET_COL,
            color_discrete_sequence=PALETTE,
            title="Longitud de descripción (palabras) vs AdoptionSpeed",
            labels={"desc_words": "Palabras"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Por tipo de mascota")
    left, right = st.columns(2)

    with left:
        fig = px.box(
            filtered, x=TARGET_COL, y="desc_len", color="Type_label",
            color_discrete_sequence=PALETTE,
            title="Longitud (caracteres) por Tipo y AdoptionSpeed",
            labels={"desc_len": "Caracteres"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            filtered.sample(min(2000, len(filtered)), random_state=42),
            x="desc_words", y=TARGET_COL, color="Type_label",
            opacity=0.4, title="Palabras en descripción vs AdoptionSpeed",
            color_discrete_sequence=PALETTE,
            trendline="ols",
            labels={"desc_words": "Palabras", TARGET_COL: "AdoptionSpeed"},
        )
        st.plotly_chart(fig, use_container_width=True)

    tbl = (
        filtered.groupby([TARGET_COL])[["desc_len", "desc_words"]]
        .mean().round(1)
    )
    st.caption("Longitud promedio de descripción por AdoptionSpeed")
    st.dataframe(tbl, use_container_width=True)


def tab_conclusiones() -> None:
    st.subheader("Conclusiones del EDA")
    st.info("Completar con las observaciones propias del análisis.")
    st.markdown("""
**Sugerencias de puntos a cubrir:**

- Distribución del target y si existe desbalance de clases
- Variables con mayor poder discriminante respecto a `AdoptionSpeed`
- Diferencias clave entre perros y gatos en el proceso de adopción
- Variables de salud: ¿las mascotas vacunadas/esterilizadas se adoptan más rápido?
- Impacto de la cantidad de fotos y la longitud de la descripción
- Outliers identificados y decisión de tratamiento
- Variables candidatas a descartar o transformar antes del modelado
""")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("🐾 EDA Final — PetFinder Adoption Prediction")
    st.write(
        "Aplicación interactiva basada en el notebook `tutoriales/03_EDA_final.ipynb`. "
        "Usá los filtros del panel izquierdo para explorar subconjuntos del dataset."
    )

    try:
        df = load_data()
    except FileNotFoundError as exc:
        st.error(f"No se encontró el archivo de datos: {exc}")
        st.stop()

    filtered = apply_filters(df)

    if filtered.empty:
        st.warning("Los filtros no devolvieron registros. Ampliá el rango o seleccioná más clases.")
        st.stop()

    tabs = st.tabs([
        "📋 Resumen",
        "🔍 Calidad",
        "🎯 Target",
        "🔢 Numéricas",
        "🏷️ Categóricas",
        "🐕🐈 Perros vs Gatos",
        "📊 Correlación",
        "📝 Texto",
        "✅ Conclusiones",
    ])

    with tabs[0]: tab_resumen(df, filtered)
    with tabs[1]: tab_calidad(df)
    with tabs[2]: tab_target(filtered)
    with tabs[3]: tab_numericas(filtered)
    with tabs[4]: tab_categoricas(filtered)
    with tabs[5]: tab_bivariado(filtered)
    with tabs[6]: tab_correlacion(filtered)
    with tabs[7]: tab_texto(filtered)
    with tabs[8]: tab_conclusiones()


if __name__ == "__main__":
    main()
