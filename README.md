# Boston Housing: lineární regrese a PCR

Projekt porovnává klasickou vícenásobnou lineární regresi s metodou
**Principal Component Regression (PCR)** na datasetu Boston Housing.

## Struktura projektu

```text
boston-housing-regression/
├── data/
│   └── BostonHousing.csv
├── src/
│   └── analysis.py
├── results/                 # vytvoří se automaticky po spuštění
├── .gitignore
├── README.md
└── requirements.txt
```

## Co skript dělá

1. načte lokální CSV a zkontroluje data;
2. vytvoří základní průzkumné grafy;
3. rozdělí data na trénovací a testovací množinu;
4. natrénuje lineární regresi;
5. pomocí 10násobné cross-validace vybere nejlepší počet komponent pro PCR;
6. porovná oba modely pomocí RMSE a R²;
7. uloží tabulku výsledků a grafy do složky `results`.

Dataset se nestahuje z internetu. Tím se odstraňuje původní chyba
`SSLCertVerificationError` a projekt funguje nezávisle na externím serveru.

## Instalace

V terminálu otevřete kořenovou složku projektu a spusťte:

```bash
python -m venv .venv
```

Aktivace prostředí ve Windows PowerShellu:

```powershell
.venv\Scripts\Activate.ps1
```

Instalace knihoven:

```bash
python -m pip install -r requirements.txt
```

## Spuštění

```bash
python src/analysis.py
```

## Interpretace metrik

- **RMSE:** nižší hodnota znamená přesnější predikce.
- **R²:** vyšší hodnota znamená, že model vysvětluje větší část variability cíle.

Výsledky na testovací množině se používají až pro závěrečné nezávislé
vyhodnocení. Počet komponent PCR se vybírá pouze pomocí cross-validace na
trénovacích datech.

## Poznámka k datasetu

Boston Housing je historický výukový dataset. Některé jeho proměnné a jejich
původní interpretace jsou dnes považovány za eticky problematické. Projekt jej
proto používá pouze pro demonstraci regresních metod a ne pro reálné rozhodování.
