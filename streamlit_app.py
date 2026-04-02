from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="EDA PetFinder",
    page_icon="📊",
    layout="wide",
)


DATA_PATH = Path("input/petfinder-adoption-prediction/train/train.csv")

TARGET_COL = "AdoptionSpeed"
NUMERIC_COLS = ["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]
CATEGORICAL_COLS = [
    "Type",
    "Breed1",
    "Breed2",
    "Gender",
    "Color1",
    "Color2",
    "Color3",
    "MaturitySize",
    "FurLength",
    "Vaccinated",
    "Dewormed",
    "Sterilized",
    "Health",
    "State",
]
TEXT_COLS = ["Name", "Description"]
ID_COLS = ["RescuerID", "PetID"]

TYPE_MAP = {1: "Dog", 2: "Cat"}
GENDER_MAP = {1: "Male", 2: "Female", 3: "Mixed"}
SIZE_MAP = {
    0: "Not specified",
    1: "Small",
    2: "Medium",
    3: "Large",
    4: "Extra large",
}
YES_NO_MAP = {1: "Yes", 2: "No", 3: "Not sure"}
HEALTH_MAP = {0: "Not specified", 1: "Healthy", 2: "Minor injury", 3: "Serious injury"}
TARGET_LABELS = {
    0: "0 - Very fast",
    1: "1 - Fast",
    2: "2 - Moderate",
    3: "3 - Slow",
    4: "4 - Not adopted / very slow",
}
AGE_BINS = [-1, 6, 12, 24, 60, float("inf")]
AGE_LABELS = ["0-6 meses", "7-12 meses", "13-24 meses", "25-60 meses", "60+ meses"]


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos en {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["TypeLabel"] = df["Type"].map(TYPE_MAP).fillna("Unknown")
    df["GenderLabel"] = df["Gender"].map(GENDER_MAP).fillna("Unknown")
    df["MaturitySizeLabel"] = df["MaturitySize"].map(SIZE_MAP).fillna("Unknown")
    df["VaccinatedLabel"] = df["Vaccinated"].map(YES_NO_MAP).fillna("Unknown")
    df["SterilizedLabel"] = df["Sterilized"].map(YES_NO_MAP).fillna("Unknown")
    df["HealthLabel"] = df["Health"].map(HEALTH_MAP).fillna("Unknown")
    df["AdoptionSpeedLabel"] = df[TARGET_COL].map(TARGET_LABELS)
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    type_options = ["Todos"] + sorted(df["TypeLabel"].dropna().unique().tolist())
    selected_type = st.sidebar.selectbox("Tipo de mascota", type_options)

    speed_options = sorted(df[TARGET_COL].dropna().unique().tolist())
    selected_speed = st.sidebar.multiselect(
        "Clases de AdoptionSpeed",
        speed_options,
        default=speed_options,
        help="0 = adopcion muy rapida, 4 = adopcion muy lenta o no adoptado.",
    )

    age_range = st.sidebar.slider(
        "Edad (meses)",
        int(df["Age"].min()),
        int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max())),
    )

    filtered = df.copy()
    if selected_type != "Todos":
        filtered = filtered[filtered["TypeLabel"] == selected_type]
    filtered = filtered[filtered[TARGET_COL].isin(selected_speed)]
    filtered = filtered[filtered["Age"].between(age_range[0], age_range[1])]
    return filtered


def section_title(title: str, caption: str) -> None:
    st.subheader(title)
    st.caption(caption)


def show_overview(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    section_title("Resumen general", "Vista rápida del dataset y de la muestra filtrada.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros totales", f"{len(df):,}")
    c2.metric("Registros filtrados", f"{len(filtered):,}")
    c3.metric("Columnas", len(df.columns))
    c4.metric("Missing en Name + Description", int(df["Name"].isna().sum() + df["Description"].isna().sum()))

    dictionary_df = pd.DataFrame(
        [
            ("Type", "Tipo de mascota"),
            ("Age", "Edad en meses"),
            ("Fee", "Tarifa de adopcion"),
            ("PhotoAmt", "Cantidad de fotos"),
            ("Vaccinated", "Estado de vacunacion"),
            ("Sterilized", "Estado de esterilizacion/castracion"),
            ("Health", "Estado de salud"),
            ("AdoptionSpeed", "Objetivo: menor valor = adopcion mas rapida"),
        ],
        columns=["Variable", "Descripcion"],
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.dataframe(filtered.head(15), use_container_width=True)
    with right:
        st.dataframe(dictionary_df, use_container_width=True, hide_index=True)


def show_quality(df: pd.DataFrame) -> None:
    section_title("Calidad de datos", "Tipos de variables y valores faltantes.")

    type_summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_unique": df.nunique(),
            "missing": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100).round(2),
        }
    ).sort_values(["missing", "n_unique"], ascending=[False, False])

    missing_df = (
        type_summary[type_summary["missing"] > 0]
        .reset_index()
        .rename(columns={"index": "variable"})
        .sort_values("missing", ascending=False)
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.dataframe(type_summary, use_container_width=True)
    with right:
        if missing_df.empty:
            st.success("No se detectaron faltantes.")
        else:
            fig = px.bar(
                missing_df,
                x="missing_pct",
                y="variable",
                orientation="h",
                title="Porcentaje de valores faltantes",
                color="missing_pct",
                color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


def show_target(filtered: pd.DataFrame) -> None:
    section_title("Variable objetivo", "Distribución de AdoptionSpeed en la muestra filtrada.")

    target_summary = (
        filtered[TARGET_COL]
        .value_counts()
        .sort_index()
        .rename_axis(TARGET_COL)
        .reset_index(name="count")
    )
    target_summary["percentage"] = (target_summary["count"] / len(filtered) * 100).round(2)
    target_summary["label"] = target_summary[TARGET_COL].map(TARGET_LABELS)

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            target_summary,
            x=TARGET_COL,
            y="count",
            text="count",
            color=TARGET_COL,
            title="Distribucion de AdoptionSpeed",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(
            target_summary,
            values="count",
            names="label",
            title="Participacion por clase",
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(target_summary[["label", "count", "percentage"]], use_container_width=True, hide_index=True)


def show_numeric(filtered: pd.DataFrame) -> None:
    section_title("Variables numericas", "Resumen, distribuciones y relacion simple con la variable objetivo.")

    st.dataframe(filtered[NUMERIC_COLS].describe().T, use_container_width=True)

    selected_numeric = st.selectbox("Variable numerica", NUMERIC_COLS, index=0)
    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            filtered,
            x=selected_numeric,
            nbins=40,
            title=f"Distribucion de {selected_numeric}",
            color_discrete_sequence=["#0f766e"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(
            filtered,
            x=TARGET_COL,
            y=selected_numeric,
            color=TARGET_COL,
            title=f"{selected_numeric} vs AdoptionSpeed",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def build_proportion_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    plot_df = (
        df.groupby(col)[TARGET_COL]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )
    plot_df["proportion_pct"] = (plot_df["proportion"] * 100).round(2)
    return plot_df


def show_categorical(filtered: pd.DataFrame) -> None:
    section_title("Variables categoricas", "Distribuciones simples y cruces con la variable objetivo.")

    selected_map = {
        "TypeLabel": "Tipo",
        "GenderLabel": "Genero",
        "MaturitySizeLabel": "Tamano adulto",
        "VaccinatedLabel": "Vacunacion",
        "SterilizedLabel": "Esterilizacion",
        "HealthLabel": "Salud",
    }
    selected_col = st.selectbox("Variable categorica", list(selected_map.keys()), format_func=selected_map.get)

    left, right = st.columns(2)
    with left:
        count_df = (
            filtered[selected_col]
            .value_counts()
            .rename_axis(selected_col)
            .reset_index(name="count")
        )
        fig = px.bar(
            count_df,
            x=selected_col,
            y="count",
            text="count",
            title=f"Distribucion de {selected_map[selected_col]}",
            color="count",
            color_continuous_scale="Purples",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        prop_df = build_proportion_df(filtered, selected_col)
        fig = px.bar(
            prop_df,
            x=selected_col,
            y="proportion",
            color=TARGET_COL,
            barmode="group",
            title=f"{selected_map[selected_col]} vs AdoptionSpeed",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


def show_bivariate(filtered: pd.DataFrame) -> None:
    section_title("Cruces relevantes", "Repite los analisis bivariados mas importantes del notebook.")

    left, right = st.columns(2)
    with left:
        type_df = build_proportion_df(filtered, "TypeLabel")
        fig = px.bar(
            type_df,
            x="TypeLabel",
            y="proportion",
            color=TARGET_COL,
            barmode="group",
            title="AdoptionSpeed segun tipo de mascota",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        age_df = build_proportion_df(filtered.dropna(subset=["AgeGroup"]), "AgeGroup")
        fig = px.bar(
            age_df,
            x="AgeGroup",
            y="proportion",
            color=TARGET_COL,
            barmode="group",
            title="AdoptionSpeed segun rango de edad",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    triple_cols = ["VaccinatedLabel", "SterilizedLabel", "HealthLabel"]
    labels = {
        "VaccinatedLabel": "Vacunacion",
        "SterilizedLabel": "Esterilizacion",
        "HealthLabel": "Salud",
    }
    cols = st.columns(3)
    for col_container, feature in zip(cols, triple_cols):
        with col_container:
            plot_df = build_proportion_df(filtered, feature)
            fig = px.bar(
                plot_df,
                x=feature,
                y="proportion",
                color=TARGET_COL,
                barmode="group",
                title=f"Adopcion segun {labels[feature].lower()}",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)


def show_correlation_outliers(filtered: pd.DataFrame) -> None:
    section_title("Correlacion y outliers", "Matriz de correlacion del subconjunto numerico y revision visual de outliers.")

    corr = filtered[NUMERIC_COLS + [TARGET_COL]].corr(numeric_only=True).round(2)
    corr_df = corr.reset_index().rename(columns={"index": "variable"})
    corr_long = corr_df.melt(id_vars="variable", var_name="correlates_with", value_name="value")

    left, right = st.columns([1.05, 1])
    with left:
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Matriz de correlacion",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        outlier_var = st.selectbox("Variable para revisar outliers", NUMERIC_COLS, index=2)
        fig = px.box(
            filtered,
            x=outlier_var,
            points="outliers",
            title=f"Outliers potenciales en {outlier_var}",
            color_discrete_sequence=["#ef4444"],
        )
        st.plotly_chart(fig, use_container_width=True)


def show_conclusions() -> None:
    section_title("Conclusiones", "Resumen de hallazgos principales del EDA original.")
    conclusions = [
        "El dataset tiene tamaño suficiente para una primera tarea de clasificacion supervisada.",
        "AdoptionSpeed tiene varias clases y no está perfectamente balanceada.",
        "Los faltantes se concentran principalmente en Name y Description.",
        "Fee, Quantity, VideoAmt y PhotoAmt muestran asimetría y posibles outliers.",
        "Muchas columnas numéricas son categorías codificadas y no deben interpretarse como continuas.",
        "Type, Gender, MaturitySize, Vaccinated, Sterilized y Health parecen aportar señal útil.",
        "No se observa correlación lineal fuerte entre las variables numéricas reales.",
        "Antes de modelar conviene definir imputación, tratamiento de categóricas, uso de texto/IDs y estrategia frente a outliers.",
    ]
    for idx, item in enumerate(conclusions, start=1):
        st.write(f"{idx}. {item}")


def main() -> None:
    st.title("EDA interactivo de PetFinder")
    st.write(
        "Aplicacion Streamlit basada en el notebook `tutoriales/03_01_EDA.ipynb` para explorar "
        "el dataset de adopcion de mascotas de forma interactiva."
    )

    try:
        df = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    filtered = apply_filters(df)
    if filtered.empty:
        st.warning("Los filtros no devolvieron registros. Probá ampliar el rango o seleccionar más clases.")
        st.stop()

    tabs = st.tabs(
        [
            "Resumen",
            "Calidad",
            "Objetivo",
            "Numericas",
            "Categoricas",
            "Cruces",
            "Correlacion",
            "Conclusiones",
        ]
    )

    with tabs[0]:
        show_overview(df, filtered)
    with tabs[1]:
        show_quality(df)
    with tabs[2]:
        show_target(filtered)
    with tabs[3]:
        show_numeric(filtered)
    with tabs[4]:
        show_categorical(filtered)
    with tabs[5]:
        show_bivariate(filtered)
    with tabs[6]:
        show_correlation_outliers(filtered)
    with tabs[7]:
        show_conclusions()


if __name__ == "__main__":
    main()
