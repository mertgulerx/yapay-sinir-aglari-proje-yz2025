#!/usr/bin/env python3
"""Veri setlerinin görselleştirilmesi — rapor için grafikler."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = '/home/mert/Desktop/weekend/yapay-zeka/report/img/data'
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})


def load_csv(name):
    d = np.loadtxt(name, delimiter=',')
    return d[:, :625], d[:, 625]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════
# 1. Çıktı dağılımları (her görev için histogram)
# ═══════════════════════════════════════════════════════════
print("1. Çıktı dağılım grafikleri...")

datasets = {
    'A': ('a_200.csv', 'Problem A — Min Manhattan Mesafesi Dağılımı', 'Mesafe'),
    'B': ('b_200.csv', 'Problem B — Max Manhattan Mesafesi Dağılımı', 'Mesafe'),
    'C': ('c_500.csv', 'Problem C — Nokta Sayısı Dağılımı', 'Nokta Sayısı'),
    'D': ('d_1000.csv', 'Problem D — Tek/Çift Dağılımı', 'Sınıf'),
    'E': ('e_200.csv', 'Problem E — Köşe Manhattan Mesafesi Dağılımı', 'Mesafe'),
}

for name, (csv, title, xlabel) in datasets.items():
    _, y = load_csv(csv)
    vals, counts = np.unique(y.astype(int), return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 4) if name not in ['C', 'D'] else (7, 4))

    if name == 'D':
        bars = ax.bar(['Çift (0)', 'Tek (1)'], counts, color=['#3498db', '#e74c3c'], width=0.5)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(c), ha='center', fontweight='bold')
    else:
        color = {'A': '#2ecc71', 'B': '#e74c3c', 'C': '#3498db', 'E': '#9b59b6'}[name]
        bars = ax.bar(vals, counts, color=color, edgecolor='white', linewidth=0.3)
        for bar, c in zip(bars, counts):
            if len(vals) <= 15:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(c), ha='center', fontsize=8)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Örnek Sayısı')
    ax.set_ylim(0, max(counts) * 1.15)

    total = len(y)
    unique_vals = len(vals)
    ax.text(0.98, 0.95, f'Toplam: {total}\nBenzersiz değer: {unique_vals}\nHer değerden: {counts[0]}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    save(fig, f'dist_{name.lower()}.jpg')


# ═══════════════════════════════════════════════════════════
# 2. Tüm görevlerin dağılımları tek grafikte
# ═══════════════════════════════════════════════════════════
print("\n2. Birleşik dağılım grafiği...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

colors = {'A': '#2ecc71', 'B': '#e74c3c', 'C': '#3498db', 'D': '#f39c12', 'E': '#9b59b6'}

for idx, (name, (csv, title, xlabel)) in enumerate(datasets.items()):
    ax = axes_flat[idx]
    _, y = load_csv(csv)
    vals, counts = np.unique(y.astype(int), return_counts=True)

    if name == 'D':
        ax.bar(['Çift (0)', 'Tek (1)'], counts, color=[colors[name]], width=0.5)
    else:
        ax.bar(vals, counts, color=colors[name], edgecolor='white', linewidth=0.3)

    ax.set_title(f'Problem {name}', fontweight='bold', fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Örnek Sayısı')
    ax.text(0.97, 0.93, f'n={len(y)}', transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

axes_flat[5].axis('off')
fig.suptitle('Tüm Problemler — Çıktı Değeri Dağılımları', fontsize=15, fontweight='bold')
plt.tight_layout()
save(fig, 'dist_all.jpg')


# ═══════════════════════════════════════════════════════════
# 3. Örnek matris görselleştirmeleri
# ═══════════════════════════════════════════════════════════
print("\n3. Örnek matris görselleştirmeleri...")

for name, (csv, title, xlabel) in datasets.items():
    X, y = load_csv(csv)

    fig, axes_row = plt.subplots(1, 5, figsize=(15, 3.5))
    fig.suptitle(f'Problem {name} — Örnek Matrisler', fontsize=13, fontweight='bold')

    # Farklı çıktı değerlerinden 5 örnek seç
    vals_unique = np.unique(y.astype(int))
    if len(vals_unique) < 5:
        # D gibi az sınıflı görevler: her sınıftan birden fazla örnek
        sample_indices = []
        per_class = 5 // len(vals_unique)
        extra = 5 % len(vals_unique)
        for vi, v in enumerate(vals_unique):
            n = per_class + (1 if vi < extra else 0)
            idxs = np.where(y.astype(int) == v)[0]
            sample_indices.extend(idxs[:n])
        sample_indices = sample_indices[:5]
    else:
        step = max(1, len(vals_unique) // 5)
        sample_vals = vals_unique[::step][:5]
        sample_indices = [np.where(y.astype(int) == sv)[0][0] for sv in sample_vals]

    for i, (ax, idx) in enumerate(zip(axes_row, sample_indices)):
        mat = X[idx].reshape(25, 25)
        sv = int(y[idx])

        ax.imshow(mat, cmap='binary', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'{xlabel}={sv}', fontsize=10)
        ax.set_xticks([0, 12, 24])
        ax.set_yticks([0, 12, 24])
        ax.tick_params(labelsize=7)

        # Nokta konumlarını kırmızı ile işaretle
        points = np.argwhere(mat == 1)
        for py, px in points:
            ax.plot(px, py, 'rs', markersize=4)

    plt.tight_layout()
    save(fig, f'samples_{name.lower()}.jpg')


# ═══════════════════════════════════════════════════════════
# 4. Veri seti özet tablosu (görsel)
# ═══════════════════════════════════════════════════════════
print("\n4. Veri seti özet tablosu...")

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

table_data = [
    ['A', 'a_200.csv', '3400', '5 (sabit)', '1-17', '200', 'Regresyon\n(Min Manhattan)'],
    ['B', 'b_200.csv', '8400', '5 (sabit)', '7-48', '200', 'Regresyon\n(Max Manhattan)'],
    ['C', 'c_500.csv', '5000', '1-10', '1-10', '500', 'Sınıflandırma\n(Nokta sayısı)'],
    ['D', 'd_1000.csv', '2000', '1-10', '0-1', '1000', 'Binary\n(Tek/Çift)'],
    ['E', 'e_200.csv', '9600', '1-10', '1-48', '200', 'Regresyon\n(Köşe mesafesi)'],
]

col_labels = ['Problem', 'Dosya', 'Toplam\nÖrnek', 'Nokta\nSayısı', 'Çıktı\nAralığı',
              'Her Değerden\nÖrnek', 'Problem Tipi']

table = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                 cellLoc='center', colColours=['#3498db']*7)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Header stilini ayarla
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#2c3e50')
    else:
        cell.set_facecolor('#ecf0f1' if row % 2 == 0 else 'white')

fig.suptitle('Veri Setleri Özet Tablosu', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
save(fig, 'dataset_summary_table.jpg')


# ═══════════════════════════════════════════════════════════
# 5. Train/Test split bilgisi
# ═══════════════════════════════════════════════════════════
print("\n5. Train/Test split tablosu...")

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis('off')

split_data = [
    ['A', '3400', '2720 (80%)', '680 (20%)', '680 (25%)', '1360 (50%)', '2720 (100%)'],
    ['B', '8400', '6720 (80%)', '1680 (20%)', '1680 (25%)', '3360 (50%)', '6720 (100%)'],
    ['C', '5000', '4000 (80%)', '1000 (20%)', '1000 (25%)', '2000 (50%)', '4000 (100%)'],
    ['D', '2000', '1600 (80%)', '400 (20%)', '400 (25%)', '800 (50%)', '1600 (100%)'],
    ['E', '9600', '7680 (80%)', '1920 (20%)', '1920 (25%)', '3840 (50%)', '7680 (100%)'],
]

col_labels = ['Problem', 'Toplam', 'Eğitim', 'Test', 'Çeyrek\nEğitim', 'Yarım\nEğitim', 'Tam\nEğitim']

table = ax.table(cellText=split_data, colLabels=col_labels, loc='center',
                 cellLoc='center', colColours=['#27ae60']*7)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#1e8449')
    else:
        cell.set_facecolor('#eafaf1' if row % 2 == 0 else 'white')

fig.suptitle('Eğitim / Test Veri Bölümü', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
save(fig, 'train_test_split.jpg')


print(f"\nToplam görseller {OUT} klasöründe kaydedildi.")
print("Dosyalar:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f} ({size/1024:.0f} KB)")
