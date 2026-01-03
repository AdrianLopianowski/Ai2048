import pandas as pd
import matplotlib.pyplot as plt
import os

# --- KONFIGURACJA ---
CSV_FILE = "training_history.csv"
WINDOW_SIZE = 50  # Wygładzanie (średnia krocząca)
DPI = 120         # Jakość obrazków

def setup_style():
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')

# 1. WYKRES WYNIKÓW (Sama średnia wyników)
def plot_just_scores(df):
    print("Generowanie: Wykres Wyników...")
    rolling_score = df['Score'].rolling(window=WINDOW_SIZE).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'Historia Średniego Wyniku (Okno: {WINDOW_SIZE})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epizody', fontsize=12)
    ax.set_ylabel('Średni Wynik (Score)', fontsize=12)

    ax.plot(df['Episode'], rolling_score, color='tab:blue', linewidth=2, label='Avg Score')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("wykres_wyniki.png", dpi=DPI)
    plt.close()

# 2. WYKRES WAG (Bez zmian)
def plot_weights(df):
    print("Generowanie: Wykres Wag...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Normal
    ax1.set_title('Wagi NORMAL (Budowanie)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Wartość', fontsize=12)
    weights_normal = {
        'N_Snake': ('green', 'Snake'), 'N_Corner': ('purple', 'Corner'),
        'N_Neigh': ('brown', 'Neighbor'), 'N_Merge': ('red', 'Merge'),
        'N_Empty': ('gray', 'Empty'), 'N_Max': ('orange', 'MaxTile')
    }
    for col, (color, label) in weights_normal.items():
        if col in df.columns:
            ax1.plot(df['Episode'], df[col], color=color, label=label, linewidth=2, alpha=0.8)
    ax1.legend(loc='upper left', ncol=2)
    ax1.grid(True)

    # Panic
    ax2.set_title('Wagi PANIC (Ratunek)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Wartość', fontsize=12)
    ax2.set_xlabel('Epizody', fontsize=12)
    weights_panic = {
        'P_Snake': ('green', 'Snake'), 'P_Corner': ('purple', 'Corner'),
        'P_Merge': ('red', 'Merge'), 'P_Empty': ('gray', 'Empty'),
        'P_Max': ('orange', 'MaxTile')
    }
    for col, (color, label) in weights_panic.items():
        if col in df.columns:
            ax2.plot(df['Episode'], df[col], color=color, label=label, linewidth=2, alpha=0.8)
    ax2.legend(loc='upper left', ncol=2)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("wykres_wagi.png", dpi=DPI)
    plt.close()

# 3. WYKRES KLOCKÓW (Dwa ploty: Max i Średni Max)
def plot_tiles_split(df):
    print("Generowanie: Wykres Klocków (2 ploty)...")

    # Dane
    rolling_peak_max = df['MaxTile'].rolling(window=WINDOW_SIZE).max()
    rolling_avg_max = df['MaxTile'].rolling(window=WINDOW_SIZE).mean()

    # Tworzymy 2 oddzielne wykresy jeden pod drugim
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot 1: REKORDOWY MAX KLOCEK
    ax1.set_title(f'Rekordowy Max Klocek (Najlepszy w oknie {WINDOW_SIZE})', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Wartość Klocka', fontsize=12)
    ax1.plot(df['Episode'], rolling_peak_max, color='tab:red', linewidth=2, label='Rekord MaxTile')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: ŚREDNI MAX KLOCEK
    ax2.set_title(f'Średni Max Klocek (Stabilność)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Wartość Klocka', fontsize=12)
    ax2.set_xlabel('Epizody', fontsize=12)
    ax2.plot(df['Episode'], rolling_avg_max, color='tab:orange', linewidth=2, label='Średni MaxTile')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("wykres_klocki.png", dpi=DPI)
    plt.close()

# 4. WYKRES RUCHÓW (Sama średnia liczba ruchów)
def plot_moves_only(df):
    print("Generowanie: Wykres Ruchów...")
    if 'Moves' not in df.columns:
        print("Brak danych o ruchach.")
        return

    rolling_moves = df['Moves'].rolling(window=WINDOW_SIZE).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'Średnia Liczba Ruchów (Przetrwanie)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epizody', fontsize=12)
    ax.set_ylabel('Liczba Ruchów', fontsize=12)

    ax.plot(df['Episode'], rolling_moves, color='purple', linewidth=2, label='Avg Moves')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("wykres_ruchy.png", dpi=DPI)
    plt.close()

def generate_charts():
    if not os.path.exists(CSV_FILE):
        print(f"Brak pliku {CSV_FILE}")
        return
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty: return
    except: return

    setup_style()

    # Generowanie 4 niezależnych zestawów
    plot_just_scores(df)      # 1. Wyniki
    plot_weights(df)          # 2. Wagi
    plot_tiles_split(df)      # 3. Klocki (Max i Średni)
    plot_moves_only(df)       # 4. Ruchy

    print("--> Zakończono generowanie wszystkich 4 wykresów.")

    try:
        if os.name == 'nt':
            os.startfile("wykres_wyniki.png")
            os.startfile("wykres_wagi.png")
            os.startfile("wykres_klocki.png")
            os.startfile("wykres_ruchy.png")
    except: pass

if __name__ == "__main__":
    generate_charts()