"""Genera los gráficos y el reporte interactivo del centro de formación.

Desde la carpeta del proyecto, ejecute:
    python python/analisis_visualizacion.py --origen csv
También se puede cargar la base de datos local:
    python python/analisis_visualizacion.py --origen sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


BASE = Path(__file__).resolve().parents[1]
GRAFICOS = BASE / "graficos"

sns.set_theme(style="whitegrid", palette="colorblind")


def cargar_datos(origen: str) -> pd.DataFrame:
    """Carga el conjunto analítico desde el CSV o la base SQLite."""
    if origen == "csv":
        ruta = BASE / "data" / "processed" / "dataset_analitico.csv"
        return pd.read_csv(ruta, parse_dates=["fecha"])

    ruta = BASE / "database" / "centro_formacion.db"
    with sqlite3.connect(ruta) as conexion:
        datos = pd.read_sql_query("SELECT * FROM dataset_analitico", conexion)
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    return datos


def guardar_figura(fig: plt.Figure, nombre: str, fuente: str) -> None:
    fig.text(
        0.99,
        0.006,
        f"Fuente: {fuente} (datos sintéticos)",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(GRAFICOS / nombre, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generar_graficos(datos: pd.DataFrame, fuente: str) -> None:
    """Construye y guarda los nueve gráficos del proyecto."""
    GRAFICOS.mkdir(parents=True, exist_ok=True)

    cursos = datos["curso"].value_counts().nlargest(5).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    cursos.plot.barh(ax=ax, color="#17365D")
    ax.set(title="Cinco cursos con más matrículas", xlabel="Matrículas", ylabel="Curso")
    guardar_figura(fig, "01_top_5_cursos.png", fuente)

    mensual = datos.groupby(datos["fecha"].dt.to_period("M")).size().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mensual.index.astype(str), mensual.values, marker="o", color="#E91E75")
    ax.set(title="Evolución mensual de las matrículas", xlabel="Mes", ylabel="Matrículas")
    guardar_figura(fig, "02_matriculas_mensuales.png", fuente)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=datos, x="nota", bins=12, kde=True, color="#17365D", ax=ejes[0])
    sns.histplot(data=datos, x="progreso", bins=12, kde=True, color="#18A6A6", ax=ejes[1])
    ejes[0].set(title="Distribución de notas", xlabel="Nota (0 a 5)", ylabel="Inscripciones")
    ejes[1].set(
        title="Distribución del progreso",
        xlabel="Progreso (%)",
        ylabel="Inscripciones",
    )
    guardar_figura(fig, "03_distribuciones_nota_progreso.png", fuente)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=datos, x="modalidad", y="nota", ax=ax)
    ax.set(
        title="Distribución de notas por modalidad",
        xlabel="Modalidad",
        ylabel="Nota (0 a 5)",
        ylim=(0, 5),
    )
    guardar_figura(fig, "04_notas_por_modalidad.png", fuente)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=datos, x="progreso", y="nota", hue="estado", alpha=0.75, ax=ax)
    ax.set(
        title="Relación entre progreso y nota",
        xlabel="Progreso (%)",
        ylabel="Nota (0 a 5)",
        ylim=(0, 5),
    )
    guardar_figura(fig, "05_progreso_vs_nota.png", fuente)

    variables = ["edad", "nivel_socioeconomico", "progreso", "nota", "satisfaccion", "precio"]
    matriz = datos[variables].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Matriz de correlaciones entre variables numéricas")
    guardar_figura(fig, "06_matriz_correlaciones.png", fuente)

    aprobacion = (
        datos.groupby(["curso", "modalidad"], observed=True)["aprobado"]
        .mean()
        .mul(100)
        .reset_index(name="tasa_aprobacion")
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=aprobacion, x="curso", y="tasa_aprobacion", hue="modalidad", ax=ax)
    ax.set(
        title="Tasa de aprobación por curso y modalidad",
        xlabel="Curso",
        ylabel="Aprobación (%)",
        ylim=(0, 100),
    )
    ax.tick_params(axis="x", labelrotation=35)
    guardar_figura(fig, "07_aprobacion_curso_modalidad.png", fuente)

    q1, q3 = datos["nota"].quantile([0.25, 0.75])
    rango_intercuartil = q3 - q1
    atipicos = datos.loc[
        (datos["nota"] < q1 - 1.5 * rango_intercuartil)
        | (datos["nota"] > q3 + 1.5 * rango_intercuartil)
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=datos, x="nota", color="#18A6A6", ax=ax)
    ax.set(
        title="Revisión de posibles notas atípicas (regla IQR)",
        xlabel="Nota (0 a 5)",
    )
    guardar_figura(fig, "08_revision_notas_atipicas.png", fuente)

    promedios = datos.groupby("curso", observed=True)["nota"].mean().nlargest(5).sort_values()
    fig, ejes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ejes[0].barh(promedios.index, promedios.values, color="#E91E75")
    ejes[0].set(
        title="Eje recortado: diferencia exagerada",
        xlabel="Nota media (0 a 5)",
        xlim=(max(0, float(promedios.min()) - 0.4), 5),
    )
    ejes[1].barh(promedios.index, promedios.values, color="#17365D")
    ejes[1].set(
        title="Versión corregida: eje desde cero",
        xlabel="Nota media (0 a 5)",
        xlim=(0, 5),
    )
    guardar_figura(fig, "09_correccion_eje_truncado.png", fuente)

    interactivo = px.scatter(
        datos,
        x="progreso",
        y="nota",
        color="estado",
        hover_data=["nombre", "curso", "modalidad"],
        title="Progreso y nota por estudiante",
        labels={
            "progreso": "Progreso (%)",
            "nota": "Nota (0 a 5)",
            "estado": "Estado",
        },
    )
    interactivo.update_yaxes(range=[0, 5])
    interactivo.write_html(
        GRAFICOS / "reporte_interactivo.html",
        include_plotlyjs=True,
    )

    print(f"Gráficos estáticos y reporte interactivo guardados en: {GRAFICOS}")
    print(f"Posibles notas atípicas por IQR: {len(atipicos)} de {len(datos)} registros.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera los gráficos del centro de formación.")
    parser.add_argument(
        "--origen",
        choices=("csv", "sqlite"),
        default="csv",
        help="Fuente de datos (por defecto: csv).",
    )
    args = parser.parse_args()

    datos = cargar_datos(args.origen)
    fuente = (
        "dataset_analitico.csv"
        if args.origen == "csv"
        else "tabla dataset_analitico de SQLite"
    )
    generar_graficos(datos, fuente)


if __name__ == "__main__":
    main()
