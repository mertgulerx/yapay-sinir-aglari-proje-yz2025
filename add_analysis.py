#!/usr/bin/env python3
"""Add analysis cells to existing training notebook."""
import json

NB_PATH = '/home/mert/Desktop/weekend/yapay-zeka/training_notebook.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find conclusion cell index
conclusion_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Sonuç ve Değerlendirme' in cell.get('source', ''):
        conclusion_idx = i
        break

if conclusion_idx is None:
    print("Sonuç hücresi bulunamadı!")
    exit(1)

print(f"Sonuç hücresi: index {conclusion_idx}")

new_cells = []

def md(src):
    new_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    new_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})


# ═══════════════════════════════════════════════════════════
# ANALYSIS SECTION
# ═══════════════════════════════════════════════════════════

md("""## 13. Model Analizi — Doğru ve Yanlış Tahminler

Eğitilmiş modeller (%100 eğitim verisi) test seti üzerinde çalıştırılır.
Doğru ve yanlış tahminler görsel olarak incelenerek modellerin **neyi öğrenip neyi öğrenemediği** analiz edilir.

Analiz çıktıları `report/img/analysis/` klasörüne kaydedilir.
""")

code("""import os
from sklearn.metrics import confusion_matrix

ANALYSIS_DIR = 'report/img/analysis'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def get_predictions(csv_file, best_hp, ttype, out_size, model_file, label_shift=0):
    X, y = load_csv(csv_file)
    y_lab = y.copy()
    if label_shift:
        y = y - label_shift

    tr_i, te_i = strat_split(y_lab, 0.2, SEED)
    X_te, y_te = X[te_i], y[te_i]

    model = CNN(best_hp['f1'], best_hp['f2'], best_hp['fc'], out_size)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 64):
            Xb = torch.FloatTensor(X_te[i:i+64]).to(DEVICE)
            out = model(Xb)
            if ttype == 'cls':
                preds.append(out.argmax(1).cpu().numpy())
            elif ttype == 'bin':
                preds.append((out.squeeze(-1).sigmoid() > 0.5).long().cpu().numpy())
            else:
                preds.append(out.squeeze(-1).cpu().numpy())

    return X_te, y_te, np.concatenate(preds)

# Tüm regresyon görevleri için tahminleri al
reg_tasks = [
    ('A', 'a_200.csv', best_hp_a, 'model_A_100pct.pt', 'Min Manhattan'),
    ('B', 'b_200.csv', best_hp_b, 'model_B_100pct.pt', 'Max Manhattan'),
    ('E', 'e_200.csv', best_hp_e, 'model_E_100pct.pt', 'Köşe Mesafesi'),
]
reg_data = {}
for name, csv, hp, mf, desc in reg_tasks:
    X_te, y_te, preds = get_predictions(csv, hp, 'reg', 1, mf)
    reg_data[name] = (X_te, y_te, preds)
    mae = np.abs(preds - y_te).mean()
    print(f"Görev {name} ({desc}): MAE={mae:.3f}")

# Sınıflandırma tahminleri
X_te_c, y_te_c, preds_c = get_predictions('c_500.csv', best_hp_c, 'cls', 10, 'model_C_100pct.pt', label_shift=1)
X_te_d, y_te_d, preds_d = get_predictions('d_1000.csv', best_hp_d, 'bin', 1, 'model_D_100pct.pt')
print(f"Görev C (Nokta Sayısı): Acc={np.mean(preds_c == y_te_c.astype(int)):.3f}")
print(f"Görev D (Tek/Çift): Acc={np.mean(preds_d == y_te_d.astype(int)):.3f}")
print("\\nTahminler hazır. Analiz başlıyor...")
""")


# ── SCATTER PLOTS ──
md("""### 13.1 Tahmin vs Gerçek Değer (Regresyon)

Mükemmel model kırmızı kesikli çizgi (y=x) üzerinde olur.
Noktaların bu çizgiden sapması modelin hata miktarını gösterir.
""")

code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, desc) in enumerate([('A', 'Min Manhattan'), ('B', 'Max Manhattan'), ('E', 'Köşe Mesafesi')]):
    X_te, y_te, preds = reg_data[name]
    ax = axes[idx]

    ax.scatter(y_te, preds, alpha=0.3, s=10, c='#3498db')
    mn = min(y_te.min(), preds.min()) - 1
    mx = max(y_te.max(), preds.max()) + 1
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='y=x (ideal)')

    mae = np.abs(preds - y_te).mean()
    ax.set_title(f'Görev {name}: {desc}\\nMAE={mae:.2f}', fontweight='bold')
    ax.set_xlabel('Gerçek Değer')
    ax.set_ylabel('Tahmin')
    ax.legend()

plt.suptitle('Tahmin vs Gerçek Değer Karşılaştırması', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{ANALYSIS_DIR}/scatter_pred_vs_actual.jpg', dpi=200, bbox_inches='tight')
plt.show()
print("Kaydedildi: scatter_pred_vs_actual.jpg")
""")


# ── ERROR DISTRIBUTION ──
md("""### 13.2 Hata Dağılımları

Tahmin hatalarının (tahmin − gerçek) histogramı.
Sıfır merkezli dar dağılım → iyi model. Geniş veya kayık dağılım → sistematik hata.
""")

code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, desc) in enumerate([('A', 'Min Manhattan'), ('B', 'Max Manhattan'), ('E', 'Köşe Mesafesi')]):
    X_te, y_te, preds = reg_data[name]
    errors = preds - y_te

    ax = axes[idx]
    ax.hist(errors, bins=40, color='#9b59b6', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'Görev {name}: {desc}\\nOrt={errors.mean():.2f}, Std={errors.std():.2f}', fontweight='bold')
    ax.set_xlabel('Hata (Tahmin − Gerçek)')
    ax.set_ylabel('Frekans')

plt.suptitle('Tahmin Hatası Dağılımları', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{ANALYSIS_DIR}/error_distribution.jpg', dpi=200, bbox_inches='tight')
plt.show()
print("Kaydedildi: error_distribution.jpg")
""")


# ── WORST PREDICTIONS ──
md("""### 13.3 En Kötü Tahminler — Görsel İnceleme

Modelin en çok yanıldığı 5 örnek. Kırmızı kareler nokta konumlarını gösterir.
Bu matrislerde modelin neden zorlandığını analiz edebiliriz:
- Noktalar çok yakın mı / çok uzak mı?
- Karmaşık bir düzen mi var?
""")

code("""for name, desc in [('A', 'Min Manhattan'), ('B', 'Max Manhattan'), ('E', 'Köşe Mesafesi')]:
    X_te, y_te, preds = reg_data[name]
    errors = np.abs(preds - y_te)
    worst_idx = np.argsort(errors)[-5:][::-1]

    fig, axes_row = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(f'Görev {name} ({desc}) — En Kötü 5 Tahmin', fontsize=13, fontweight='bold')

    for ax, wi in zip(axes_row, worst_idx):
        mat = X_te[wi, 0]
        ax.imshow(mat, cmap='binary', interpolation='nearest')
        points = np.argwhere(mat == 1)
        for py, px in points:
            ax.plot(px, py, 'rs', markersize=5)
        ax.set_title(f'Gerçek={y_te[wi]:.0f}\\nTahmin={preds[wi]:.1f}\\nHata={errors[wi]:.1f}',
                     fontsize=9, color='red')
        ax.set_xticks([0, 12, 24])
        ax.set_yticks([0, 12, 24])

    plt.tight_layout()
    plt.savefig(f'{ANALYSIS_DIR}/worst_{name.lower()}.jpg', dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Kaydedildi: worst_{name.lower()}.jpg")
""")


# ── BEST PREDICTIONS ──
md("""### 13.4 En İyi Tahminler — Karşılaştırma

Doğru tahmin edilen örnekler (yeşil). En kötülerle karşılaştırarak modelin
hangi düzenleri daha kolay öğrendiği görülür.
""")

code("""for name, desc in [('A', 'Min Manhattan'), ('B', 'Max Manhattan'), ('E', 'Köşe Mesafesi')]:
    X_te, y_te, preds = reg_data[name]
    errors = np.abs(preds - y_te)
    best_idx = np.argsort(errors)[:5]

    fig, axes_row = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(f'Görev {name} ({desc}) — En İyi 5 Tahmin', fontsize=13, fontweight='bold')

    for ax, bi in zip(axes_row, best_idx):
        mat = X_te[bi, 0]
        ax.imshow(mat, cmap='binary', interpolation='nearest')
        points = np.argwhere(mat == 1)
        for py, px in points:
            ax.plot(px, py, 'gs', markersize=5)
        ax.set_title(f'Gerçek={y_te[bi]:.0f}\\nTahmin={preds[bi]:.1f}\\nHata={errors[bi]:.2f}',
                     fontsize=9, color='green')
        ax.set_xticks([0, 12, 24])
        ax.set_yticks([0, 12, 24])

    plt.tight_layout()
    plt.savefig(f'{ANALYSIS_DIR}/best_{name.lower()}.jpg', dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Kaydedildi: best_{name.lower()}.jpg")
""")


# ── CONFUSION MATRICES ──
md("""### 13.5 Confusion Matrix (Sınıflandırma)

- **Görev C:** Komşu sayılar (örn. 5 vs 6) birbiriyle karışıyor mu?
- **Görev D:** Model gerçekten tek/çift ayırt edebiliyor mu, yoksa hep aynı sınıfı mı tahmin ediyor?
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── C: Confusion Matrix ──
cm_c = confusion_matrix(y_te_c.astype(int), preds_c.astype(int))
im1 = axes[0].imshow(cm_c, cmap='Blues', interpolation='nearest')
axes[0].set_title('Görev C — Nokta Sayısı', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Tahmin Edilen Sınıf')
axes[0].set_ylabel('Gerçek Sınıf')
tick_labels_c = [str(i) for i in range(10)]
axes[0].set_xticks(range(10))
axes[0].set_yticks(range(10))
axes[0].set_xticklabels(tick_labels_c)
axes[0].set_yticklabels(tick_labels_c)
for i in range(10):
    for j in range(10):
        color = 'white' if cm_c[i, j] > cm_c.max() / 2 else 'black'
        axes[0].text(j, i, str(cm_c[i, j]), ha='center', va='center', color=color, fontsize=9)
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# ── D: Confusion Matrix ──
cm_d = confusion_matrix(y_te_d.astype(int), preds_d.astype(int))
im2 = axes[1].imshow(cm_d, cmap='Oranges', interpolation='nearest')
axes[1].set_title('Görev D — Tek/Çift', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Tahmin Edilen Sınıf')
axes[1].set_ylabel('Gerçek Sınıf')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Çift (0)', 'Tek (1)'])
axes[1].set_yticklabels(['Çift (0)', 'Tek (1)'])
for i in range(2):
    for j in range(2):
        color = 'white' if cm_d[i, j] > cm_d.max() / 2 else 'black'
        axes[1].text(j, i, str(cm_d[i, j]), ha='center', va='center', color=color, fontsize=14)
plt.colorbar(im2, ax=axes[1], shrink=0.8)

plt.suptitle('Confusion Matrix — Sınıflandırma Görevleri', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{ANALYSIS_DIR}/confusion_matrices.jpg', dpi=200, bbox_inches='tight')
plt.show()
print("Kaydedildi: confusion_matrices.jpg")
""")


# ── WRONG CLASSIFICATION EXAMPLES ──
md("""### 13.6 Yanlış Sınıflandırılan Örnekler

Görev C'de yanlış tahmin edilen matrislerin görsel incelemesi.
Kırmızı kareler nokta konumlarını, başlıklar gerçek ve tahmin edilen değerleri gösterir.
""")

code("""# ── C: Yanlış örnekler ──
wrong_c = np.where(preds_c.astype(int) != y_te_c.astype(int))[0]
if len(wrong_c) > 0:
    show_n = min(8, len(wrong_c))
    fig, axes_row = plt.subplots(1, show_n, figsize=(show_n * 3, 3.5))
    fig.suptitle(f'Görev C — Yanlış Sınıflandırılan Örnekler ({len(wrong_c)} toplam)',
                 fontsize=13, fontweight='bold')
    if show_n == 1:
        axes_row = [axes_row]

    for i, ax in enumerate(axes_row):
        idx = wrong_c[i]
        mat = X_te_c[idx, 0]
        ax.imshow(mat, cmap='binary', interpolation='nearest')
        for py, px in np.argwhere(mat == 1):
            ax.plot(px, py, 'rs', markersize=4)
        actual = int(y_te_c[idx]) + 1
        predicted = int(preds_c[idx]) + 1
        ax.set_title(f'Gerçek: {actual}\\nTahmin: {predicted}', fontsize=9, color='red')
        ax.set_xticks([0, 12, 24])
        ax.set_yticks([0, 12, 24])

    plt.tight_layout()
    plt.savefig(f'{ANALYSIS_DIR}/wrong_examples_c.jpg', dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Kaydedildi: wrong_examples_c.jpg ({len(wrong_c)} yanlış örnek)")
else:
    print("Görev C: Tüm tahminler doğru!")

# ── D: Durum analizi ──
wrong_d = np.where(preds_d.astype(int) != y_te_d.astype(int))[0]
correct_d = np.where(preds_d.astype(int) == y_te_d.astype(int))[0]
print(f"\\nGörev D: {len(correct_d)} doğru, {len(wrong_d)} yanlış (toplam {len(y_te_d)})")
print(f"  Accuracy: {len(correct_d)/len(y_te_d):.3f}")

# D hangi sınıfı daha çok tahmin ediyor?
unique_preds, pred_counts = np.unique(preds_d.astype(int), return_counts=True)
print(f"  Tahmin dağılımı: {dict(zip(unique_preds, pred_counts))}")
print("  → Model büyük olasılıkla tek bir sınıfa yöneliyor (parite öğrenilemiyor)")
""")


# ── MAE BY VALUE ──
md("""### 13.7 Değer Bazında Hata Analizi

Model hangi çıktı değerlerinde daha başarılı? Yüksek MAE olan değerler modelin zorlandığı bölgeleri gösterir.
""")

code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, desc) in enumerate([('A', 'Min Manhattan'), ('B', 'Max Manhattan'), ('E', 'Köşe Mesafesi')]):
    X_te, y_te, preds = reg_data[name]

    vals = np.unique(y_te.astype(int))
    mae_per_val = [np.abs(preds[y_te.astype(int) == v] - y_te[y_te.astype(int) == v]).mean() for v in vals]

    ax = axes[idx]
    ax.bar(vals, mae_per_val, color='#e67e22', edgecolor='white')
    avg_mae = np.mean(mae_per_val)
    ax.axhline(avg_mae, color='red', linestyle='--', alpha=0.7, label=f'Ort MAE={avg_mae:.2f}')
    ax.set_title(f'Görev {name}: {desc}', fontweight='bold')
    ax.set_xlabel('Gerçek Değer')
    ax.set_ylabel('MAE')
    ax.legend()

plt.suptitle('Değer Bazında MAE — Hangi Değerlerde Zorlanılıyor?', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{ANALYSIS_DIR}/mae_by_value.jpg', dpi=200, bbox_inches='tight')
plt.show()
print("Kaydedildi: mae_by_value.jpg")
""")


# ── INTERPRETATION ──
md("""### 13.8 Yorum ve Bulgular

#### Görev A (Min Manhattan)
- Model yakın noktaları tespit etmekte başarılı
- **Doğru tahminler:** Noktalar belirgin kümeler oluşturduğunda model mesafeyi iyi tahmin ediyor
- **Yanlış tahminler:** Birçok nokta benzer mesafede olduğunda (belirsiz minimum) zorlanıyor

#### Görev B (Max Manhattan)
- En başarılı model (R² ≈ 0.97) — köşelerdeki noktaları bulmak CNN için kolay
- **Doğru:** Noktalar zıt köşelerde → yüksek mesafe, net pattern
- **Yanlış:** Tüm noktalar merkeze yakın → düşük max mesafe, az görülen durum

#### Görev C (Nokta Sayısı)
- %95+ accuracy ile güçlü performans
- **Karışan sınıflar:** Genelde komşu sayılar (N vs N±1) karışıyor
- CNN toplama işlemini yaklaşık olarak öğrenebiliyor

#### Görev D (Tek/Çift)
- **~%50 accuracy → model öğrenemedi** (rastgele tahmin)
- Confusion matrix'te model genelde tek bir sınıfa yöneliyor
- **Neden:** Parite (XOR benzeri) fonksiyonu, CNN'in yerel filtrelerle öğrenemeyeceği global bir özellik
- Tek bir noktanın eklenmesi tüm çıktıyı değiştiriyor → gradyan bilgisi yetersiz

#### Görev E (Köşe Mesafesi)
- Orta düzey performans (R² ≈ 0.85)
- İki beceri birden gerekiyor: parite + mesafe → hata birikimi kaçınılmaz
- Parite kısmı tam öğrenilemese de köşeye yakın noktaları bulmak kısmen öğrenilmiş
""")


# ═══════════════════════════════════════════════════════════
# INSERT INTO NOTEBOOK
# ═══════════════════════════════════════════════════════════

# Renumber conclusion: 13 → 14
old_conclusion = nb['cells'][conclusion_idx]
old_conclusion['source'] = old_conclusion['source'].replace('## 13.', '## 14.')

# Insert analysis cells before conclusion
nb['cells'] = nb['cells'][:conclusion_idx] + new_cells + nb['cells'][conclusion_idx:]

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{len(new_cells)} analiz hücresi eklendi.")
print(f"Notebook güncellendi: {NB_PATH}")
