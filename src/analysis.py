"""Porovnání lineární regrese a Principal Component Regression (PCR).

Projekt používá lokální soubor data/BostonHousing.csv, takže není závislý
na dostupnosti externího serveru.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "BostonHousing.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_data(path: Path) -> pd.DataFrame:
    """Načte dataset a sjednotí názvy sloupců na velká písmena."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset nebyl nalezen: {path}\n"
            "Zkontrolujte, že BostonHousing.csv leží ve složce data."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()

    if "MEDV" not in df.columns:
        raise ValueError("Dataset musí obsahovat cílový sloupec MEDV.")
    if df.isna().any().any():
        raise ValueError("Dataset obsahuje chybějící hodnoty. Nejdříve je ošetřete.")

    return df


def save_eda_plots(df: pd.DataFrame) -> None:
    """Vytvoří základní grafy pro průzkumnou analýzu dat."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["MEDV"], kde=True, bins=30, color="darkgreen")
    plt.xlabel("MEDV (tisíce USD)")
    plt.ylabel("Počet pozorování")
    plt.title("Distribuce cílové proměnné MEDV")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_medv_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Korelační matice")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_correlation_matrix.png", dpi=160)
    plt.close()

    selected_vars = ["RM", "MEDV", "AGE", "DIS", "LSTAT"]
    pair_grid = sns.pairplot(df[selected_vars], corner=True)
    pair_grid.fig.suptitle("Párové vztahy vybraných proměnných", y=1.02)
    pair_grid.savefig(RESULTS_DIR / "03_pairplot.png", dpi=160, bbox_inches="tight")
    plt.close(pair_grid.fig)


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    """Vrátí RMSE a R²."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2


def main() -> None:
    # 1. Načtení a kontrola dat
    df = load_data(DATA_PATH)
    print(f"Rozměr datasetu: {df.shape}")
    print(f"Počet chybějících hodnot: {int(df.isna().sum().sum())}")
    print("\nZákladní statistiky:")
    print(df.describe().T.round(3))

    # 2. Průzkumná analýza
    save_eda_plots(df)

    # 3. Rozdělení na vstupní a cílovou proměnnou a na train/test data
    X = df.drop(columns="MEDV")
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    # 4. Klasická lineární regrese
    linear_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    linear_cv_rmse = -cross_val_score(
        linear_model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
    )
    linear_model.fit(X_train, y_train)

    # 5. PCR: počet komponent se vybírá cross-validací pouze z train dat
    pcr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("regressor", LinearRegression()),
        ]
    )
    component_counts = list(range(1, X_train.shape[1] + 1))
    pcr_search = GridSearchCV(
        estimator=pcr_pipeline,
        param_grid={"pca__n_components": component_counts},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    pcr_search.fit(X_train, y_train)
    pcr_model = pcr_search.best_estimator_
    optimal_components = int(pcr_search.best_params_["pca__n_components"])

    # 6. Vyhodnocení na dosud nepoužitých testovacích datech
    linear_train_pred = linear_model.predict(X_train)
    linear_test_pred = linear_model.predict(X_test)
    pcr_train_pred = pcr_model.predict(X_train)
    pcr_test_pred = pcr_model.predict(X_test)

    linear_train_rmse, linear_train_r2 = evaluate_model(y_train, linear_train_pred)
    linear_test_rmse, linear_test_r2 = evaluate_model(y_test, linear_test_pred)
    pcr_train_rmse, pcr_train_r2 = evaluate_model(y_train, pcr_train_pred)
    pcr_test_rmse, pcr_test_r2 = evaluate_model(y_test, pcr_test_pred)

    results = pd.DataFrame(
        {
            "model": ["Linear Regression", "PCR"],
            "train_rmse": [linear_train_rmse, pcr_train_rmse],
            "test_rmse": [linear_test_rmse, pcr_test_rmse],
            "train_r2": [linear_train_r2, pcr_train_r2],
            "test_r2": [linear_test_r2, pcr_test_r2],
            "cv_rmse": [linear_cv_rmse.mean(), -pcr_search.best_score_],
            "pca_components": [np.nan, optimal_components],
        }
    )
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\nPorovnání modelů:")
    print(results.round(3).to_string(index=False))
    print(f"\nOptimální počet komponent PCR podle cross-validace: {optimal_components}")

    # 7. Graf RMSE při různém počtu PCA komponent
    cv_results = pd.DataFrame(pcr_search.cv_results_)
    pcr_cv_rmse = -cv_results["mean_test_score"]
    plt.figure(figsize=(8, 5))
    plt.plot(component_counts, pcr_cv_rmse, marker="o")
    plt.axvline(
        optimal_components,
        color="red",
        linestyle="--",
        label=f"Optimum: {optimal_components}",
    )
    plt.xlabel("Počet hlavních komponent")
    plt.ylabel("Průměrné CV RMSE")
    plt.title("Výběr počtu komponent pro PCR")
    plt.xticks(component_counts)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_pcr_cross_validation.png", dpi=160)
    plt.close()

    # 8. Kumulativní vysvětlená variance PCA (počítaná pouze z train dat)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca_check = PCA().fit(X_train_scaled)
    cumulative_variance = np.cumsum(pca_check.explained_variance_ratio_) * 100

    plt.figure(figsize=(8, 5))
    plt.plot(component_counts, cumulative_variance, marker="o")
    plt.axhline(80, color="red", linestyle="--", label="80 % variance")
    plt.xlabel("Počet hlavních komponent")
    plt.ylabel("Kumulativní vysvětlená variance (%)")
    plt.title("Vysvětlená variance PCA")
    plt.xticks(component_counts)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "05_explained_variance.png", dpi=160)
    plt.close()

    # 9. Predikované vs. skutečné hodnoty na testovacích datech
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    plot_data = [
        ("Lineární regrese", linear_test_pred, linear_test_r2),
        (f"PCR ({optimal_components} komponent)", pcr_test_pred, pcr_test_r2),
    ]
    limits = [min(y_test.min(), linear_test_pred.min(), pcr_test_pred.min()),
              max(y_test.max(), linear_test_pred.max(), pcr_test_pred.max())]

    for axis, (title, predictions, r2_value) in zip(axes, plot_data):
        axis.scatter(y_test, predictions, alpha=0.75)
        axis.plot(limits, limits, "r--", label="Ideální predikce")
        axis.set_title(f"{title}\nR² = {r2_value:.3f}")
        axis.set_xlabel("Skutečné MEDV")
        axis.set_ylabel("Predikované MEDV")
        axis.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "06_predictions_comparison.png", dpi=160)
    plt.close()

    # Vyšší R² a nižší RMSE znamenají lepší výsledek.
    if np.isclose(linear_test_rmse, pcr_test_rmse, atol=1e-10) and np.isclose(
        linear_test_r2, pcr_test_r2, atol=1e-10
    ):
        print("\nOba modely mají na testovacích datech shodný výkon.")
        if optimal_components == X_train.shape[1]:
            print(
                "PCR použilo všechny komponenty, a proto je matematicky "
                "ekvivalentní lineární regresi."
            )
    elif pcr_test_rmse < linear_test_rmse:
        print("\nPodle testovacího RMSE je lepší model: PCR.")
    else:
        print("\nPodle testovacího RMSE je lepší model: Linear Regression.")
    print(f"Výsledky a grafy byly uloženy do: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
