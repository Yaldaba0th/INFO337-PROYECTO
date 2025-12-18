# Script 02: Bootstrap Comparacion de Grupos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp, f_oneway, shapiro
import warnings
warnings.filterwarnings('ignore')

# Configuracion
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("="*70)
print("Script 02: Bootstrap Comparacion de Grupos")
print("="*70)
print()

print("1. Cargando Datos...")
print("-"*70)

df = pd.read_csv('data_processed_with_categories.csv')

# Verificar/crear categorias
if 'categoria' not in df.columns:
    tercil_33 = df['df_f0'].quantile(0.333)
    tercil_67 = df['df_f0'].quantile(0.667)
    
    def categorizar(x):
        if x < tercil_33:
            return 'Baja'
        elif x < tercil_67:
            return 'Media'
        else:
            return 'Alta'
    
    df['categoria'] = df['df_f0'].apply(categorizar)

lisosomas = df['particle'].unique()
n_lisosomas = len(lisosomas)

print(f"Datos cargados: {len(df)} observaciones")
print(f"Numero de lisosomas: {n_lisosomas}")
print()

# Calcular medias y diferencias observadas
media_baja = df[df['categoria']=='Baja']['avg_state2'].mean()
media_media = df[df['categoria']=='Media']['avg_state2'].mean()
media_alta = df[df['categoria']=='Alta']['avg_state2'].mean()

diff_alto_bajo_obs = media_alta - media_baja
diff_alto_medio_obs = media_alta - media_media
diff_medio_bajo_obs = media_media - media_baja

print("MEDIAS OBSERVADAS:")
print(f"  Baja:  {media_baja:.4f}")
print(f"  Media: {media_media:.4f}")
print(f"  Alta:  {media_alta:.4f}")
print()

print("DIFERENCIAS OBSERVADAS:")
print(f"  Alta - Baja:  {diff_alto_bajo_obs:.4f}")
print(f"  Alta - Media: {diff_alto_medio_obs:.4f}")
print(f"  Media - Baja: {diff_medio_bajo_obs:.4f}")
print()

print("2. Ejecutando Bootstrap por bloques...")
print("-"*70)

n_bootstrap = 10000
print(f"Iteraciones: {n_bootstrap}")
print(f"Remuestreando {n_lisosomas} lisosomas con reemplazo...")
print()

# Almacenar resultados
bootstrap_medias_baja = []
bootstrap_medias_media = []
bootstrap_medias_alta = []

bootstrap_diffs_alto_bajo = []
bootstrap_diffs_alto_medio = []
bootstrap_diffs_medio_bajo = []

# Progress bar simple
print("Progreso: ", end='', flush=True)
for i in range(n_bootstrap):
    if i % 1000 == 0:
        print(f"{i}...", end='', flush=True)
    
    # Remuestrear lisosomas
    lisosomas_boot = np.random.choice(lisosomas, size=n_lisosomas, replace=True)
    
    # Construir dataset bootstrap
    df_boot = pd.concat([df[df['particle']==lid] for lid in lisosomas_boot], 
                        ignore_index=True)
    
    # Calcular medias por grupo
    media_baja_boot = df_boot[df_boot['categoria']=='Baja']['avg_state2'].mean()
    media_media_boot = df_boot[df_boot['categoria']=='Media']['avg_state2'].mean()
    media_alta_boot = df_boot[df_boot['categoria']=='Alta']['avg_state2'].mean()
    
    # Guardar medias
    bootstrap_medias_baja.append(media_baja_boot)
    bootstrap_medias_media.append(media_media_boot)
    bootstrap_medias_alta.append(media_alta_boot)
    
    # Calcular y guardar diferencias
    bootstrap_diffs_alto_bajo.append(media_alta_boot - media_baja_boot)
    bootstrap_diffs_alto_medio.append(media_alta_boot - media_media_boot)
    bootstrap_diffs_medio_bajo.append(media_media_boot - media_baja_boot)

print("Completado")
print()

# Convertir a arrays
bootstrap_medias_baja = np.array(bootstrap_medias_baja)
bootstrap_medias_media = np.array(bootstrap_medias_media)
bootstrap_medias_alta = np.array(bootstrap_medias_alta)

bootstrap_diffs_alto_bajo = np.array(bootstrap_diffs_alto_bajo)
bootstrap_diffs_alto_medio = np.array(bootstrap_diffs_alto_medio)
bootstrap_diffs_medio_bajo = np.array(bootstrap_diffs_medio_bajo)

print("3. Calculando Intervalos de Confianza...")
print("-"*70)

# IC 95%
ic95_alto_bajo = np.percentile(bootstrap_diffs_alto_bajo, [2.5, 97.5])
ic95_alto_medio = np.percentile(bootstrap_diffs_alto_medio, [2.5, 97.5])
ic95_medio_bajo = np.percentile(bootstrap_diffs_medio_bajo, [2.5, 97.5])

# IC 98.33% (Bonferroni para 3 comparaciones)
ic9833_alto_bajo = np.percentile(bootstrap_diffs_alto_bajo, [0.835, 99.165])
ic9833_alto_medio = np.percentile(bootstrap_diffs_alto_medio, [0.835, 99.165])
ic9833_medio_bajo = np.percentile(bootstrap_diffs_medio_bajo, [0.835, 99.165])

print("IC 95%:")
print(f"  Alta - Baja:  [{ic95_alto_bajo[0]:.4f}, {ic95_alto_bajo[1]:.4f}]")
print(f"  Alta - Media: [{ic95_alto_medio[0]:.4f}, {ic95_alto_medio[1]:.4f}]")
print(f"  Media - Baja: [{ic95_medio_bajo[0]:.4f}, {ic95_medio_bajo[1]:.4f}]")
print()

print("IC 98.33% (Bonferroni):")
print(f"  Alta - Baja:  [{ic9833_alto_bajo[0]:.4f}, {ic9833_alto_bajo[1]:.4f}]")
print(f"  Alta - Media: [{ic9833_alto_medio[0]:.4f}, {ic9833_alto_medio[1]:.4f}]")
print(f"  Media - Baja: [{ic9833_medio_bajo[0]:.4f}, {ic9833_medio_bajo[1]:.4f}]")
print()

# Calcular p-valores bootstrap
p_alto_bajo = 2 * np.sum(bootstrap_diffs_alto_bajo <= 0) / n_bootstrap
p_alto_medio = 2 * np.sum(bootstrap_diffs_alto_medio <= 0) / n_bootstrap
p_medio_bajo = 2 * np.sum(bootstrap_diffs_medio_bajo <= 0) / n_bootstrap

p_alto_bajo_str = f"< {1/n_bootstrap:.4f}" if p_alto_bajo == 0 else f"{p_alto_bajo:.4f}"
p_alto_medio_str = f"< {1/n_bootstrap:.4f}" if p_alto_medio == 0 else f"{p_alto_medio:.4f}"
p_medio_bajo_str = f"< {1/n_bootstrap:.4f}" if p_medio_bajo == 0 else f"{p_medio_bajo:.4f}"

print("P-valores bootstrap:")
print(f"  Alta - Baja:  p {p_alto_bajo_str}")
print(f"  Alta - Media: p {p_alto_medio_str}")
print(f"  Media - Baja: p {p_medio_bajo_str}")
print()


print("4. Verificando Normalidad (Shapiro-Wilk)...")
print("-"*70)
print()

# Shapiro-Wilk sobre medias de grupos
print("a) Normalidad de distribuciones bootstrap de medias por grupo:")
print("-"*60)

sample_size = min(5000, n_bootstrap)

for nombre, datos in [('Baja', bootstrap_medias_baja),
                       ('Media', bootstrap_medias_media),
                       ('Alta', bootstrap_medias_alta)]:
    sample = np.random.choice(datos, size=sample_size, replace=False)
    W, p = shapiro(sample)
    print(f"Bootstrap medias {nombre}:")
    print(f"  W-statistic: {W:.4f}")
    print(f"  p-valor: {p:.4e}")
    print(f"  Interpretacion: {'Aproximadamente normal (W >= 0.95)' if W >= 0.95 else 'Desviacion de normalidad'}")
    print()

# Shapiro-Wilk sobre diferencias
print("b) Normalidad de distribuciones bootstrap de diferencias:")
print("-"*60)

for nombre, datos in [('Alta - Baja', bootstrap_diffs_alto_bajo),
                       ('Alta - Media', bootstrap_diffs_alto_medio),
                       ('Media - Baja', bootstrap_diffs_medio_bajo)]:
    sample = np.random.choice(datos, size=sample_size, replace=False)
    W, p = shapiro(sample)
    print(f"{nombre}:")
    print(f"  W-statistic: {W:.4f}")
    print(f"  p-valor: {p:.4e}")
    print(f"  Interpretacion: {'Aproximadamente normal (W >= 0.95)' if W >= 0.95 else 'Desviacion de normalidad'}")
    print()


print("5. ANOVA sobre Grupos...")
print("-"*70)
print()

print("Hipotesis:")
print("  H0: media_baja = media_media = media_alta")
print("  H1: Al menos una media de grupo es diferente")
print()

# ANOVA sobre las 3 distribuciones bootstrap de medias
f_stat_omnibus, p_anova_omnibus = f_oneway(bootstrap_medias_baja,
                                            bootstrap_medias_media,
                                            bootstrap_medias_alta)

print(f"F-statistic (omnibus): {f_stat_omnibus:.2f}")
print(f"p-valor: {p_anova_omnibus:.4e}")
print()

if p_anova_omnibus < 0.05:
    print("DECISION: Rechazamos H0 (p < 0.05)")
    print("  Existen diferencias globales significativas entre los 3 grupos.")
else:
    print("DECISION: No rechazamos H0 (p >= 0.05)")
    print("  No hay evidencia de diferencias globales.")
print()

if p_anova_omnibus < 0.05:
    print("6. T-tests...")
    print("-"*70)
    print()
    
    print("Correccion de Bonferroni:")
    print("  alfa original: 0.05")
    print("  Comparaciones: 3")
    print("  alfa ajustado: 0.05 / 3 = 0.0167")
    print()
    
    # T-test para cada comparacion
    comparaciones_datos = [
        ('Alta - Baja', bootstrap_diffs_alto_bajo),
        ('Alta - Media', bootstrap_diffs_alto_medio),
        ('Media - Baja', bootstrap_diffs_medio_bajo)
    ]
    
    resultados_ttests = []
    for nombre, datos in comparaciones_datos:
        t_stat, p_ttest = ttest_1samp(datos, 0)
        media_boot = np.mean(datos)
        se_boot = np.std(datos, ddof=1)
        
        print(f"{nombre}:")
        print(f"  Media bootstrap: {media_boot:.4f}")
        print(f"  SE bootstrap: {se_boot:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-valor: {p_ttest:.4e}")
        
        significativo = "Si" if p_ttest < 0.0167 else "No"
        print(f"  Significativo (Bonferroni): {significativo}")
        print()
        
        resultados_ttests.append({
            'Comparacion': nombre,
            'Media': media_boot,
            'SE': se_boot,
            't_stat': t_stat,
            'p_valor': p_ttest,
            'Significativo': significativo
        })
    
    print()


print("7. Guardando Resultados...")
print("-"*70)

with open('02_output_bootstrap_grupos.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("BOOTSTRAP COMPARACION DE GRUPOS - RESULTADOS COMPLETOS\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. PARAMETROS DEL ANALISIS\n")
    f.write("-"*70 + "\n")
    f.write(f"  Iteraciones bootstrap: {n_bootstrap}\n")
    f.write(f"  Unidades remuestreadas: {n_lisosomas} lisosomas\n")
    f.write(f"  Metodo: Bootstrap por bloques\n\n")
    
    f.write("2. MEDIAS OBSERVADAS\n")
    f.write("-"*70 + "\n")
    f.write(f"  Baja:  {media_baja:.4f}\n")
    f.write(f"  Media: {media_media:.4f}\n")
    f.write(f"  Alta:  {media_alta:.4f}\n\n")
    
    f.write("3. DIFERENCIAS OBSERVADAS\n")
    f.write("-"*70 + "\n")
    f.write(f"  Alta - Baja:  {diff_alto_bajo_obs:.4f}\n")
    f.write(f"  Alta - Media: {diff_alto_medio_obs:.4f}\n")
    f.write(f"  Media - Baja: {diff_medio_bajo_obs:.4f}\n\n")
    
    f.write("4. INTERVALOS DE CONFIANZA\n")
    f.write("-"*70 + "\n")
    f.write("IC 95%:\n")
    f.write(f"  Alta - Baja:  [{ic95_alto_bajo[0]:.4f}, {ic95_alto_bajo[1]:.4f}]\n")
    f.write(f"  Alta - Media: [{ic95_alto_medio[0]:.4f}, {ic95_alto_medio[1]:.4f}]\n")
    f.write(f"  Media - Baja: [{ic95_medio_bajo[0]:.4f}, {ic95_medio_bajo[1]:.4f}]\n\n")
    
    f.write("IC 98.33% (Bonferroni):\n")
    f.write(f"  Alta - Baja:  [{ic9833_alto_bajo[0]:.4f}, {ic9833_alto_bajo[1]:.4f}]\n")
    f.write(f"  Alta - Media: [{ic9833_alto_medio[0]:.4f}, {ic9833_alto_medio[1]:.4f}]\n")
    f.write(f"  Media - Baja: [{ic9833_medio_bajo[0]:.4f}, {ic9833_medio_bajo[1]:.4f}]\n\n")
    
    f.write("5. ANOVA OMNIBUS\n")
    f.write("-"*70 + "\n")
    f.write(f"  F-statistic: {f_stat_omnibus:.2f}\n")
    f.write(f"  p-valor: {p_anova_omnibus:.4e}\n")
    f.write(f"  Decision: {'Rechaza H0' if p_anova_omnibus < 0.05 else 'No rechaza H0'}\n\n")
    
    if p_anova_omnibus < 0.05:
        f.write("6. T-TESTS POST-HOC\n")
        f.write("-"*70 + "\n")
        for res in resultados_ttests:
            f.write(f"  {res['Comparacion']}:\n")
            f.write(f"    t = {res['t_stat']:.4f}, p = {res['p_valor']:.4e}\n")
            f.write(f"    Significativo (Bonferroni alfa=0.0167): {res['Significativo']}\n\n")

print("Resultados guardados en: 02_output_bootstrap_grupos.txt")
print()

print("8. Generando Visualizaciones...")
print("-"*70)

# Q-Q Plots de normalidad
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Q-Q plots de medias por grupo
for idx, (nombre, datos) in enumerate([('Baja', bootstrap_medias_baja),
                                        ('Media', bootstrap_medias_media),
                                        ('Alta', bootstrap_medias_alta)]):
    ax = axes[0, idx]
    stats.probplot(datos, dist="norm", plot=ax)
    ax.set_title(f'Q-Q plot: Bootstrap medias {nombre}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Q-Q plots de diferencias
for idx, (nombre, datos) in enumerate([('Alta - Baja', bootstrap_diffs_alto_bajo),
                                        ('Alta - Media', bootstrap_diffs_alto_medio),
                                        ('Media - Baja', bootstrap_diffs_medio_bajo)]):
    ax = axes[1, idx]
    stats.probplot(datos, dist="norm", plot=ax)
    ax.set_title(f'Q-Q plot: Diferencias {nombre}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Verificacion de Normalidad - Q-Q Plots', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_qqplots_normalidad.png', dpi=300, bbox_inches='tight')
print("Q-Q plots guardados en: 02_qqplots_normalidad.png")

# Histogramas de distribuciones bootstrap
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Histogramas de medias por grupo
for idx, (nombre, datos, color) in enumerate([('Baja', bootstrap_medias_baja, 'blue'),
                                                ('Media', bootstrap_medias_media, 'orange'),
                                                ('Alta', bootstrap_medias_alta, 'green')]):
    ax = axes[0, idx]
    ax.hist(datos, bins=50, alpha=0.7, color=color, edgecolor='black')
    ax.axvline(datos.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Media: {datos.mean():.4f}')
    ax.set_xlabel('avg_state2')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Bootstrap medias {nombre}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Histogramas de diferencias
for idx, (nombre, datos) in enumerate([('Alta - Baja', bootstrap_diffs_alto_bajo),
                                        ('Alta - Media', bootstrap_diffs_alto_medio),
                                        ('Media - Baja', bootstrap_diffs_medio_bajo)]):
    ax = axes[1, idx]
    ax.hist(datos, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(datos.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Media: {datos.mean():.4f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=2, label='H0: diff = 0')
    
    # IC 98.33%
    if idx == 0:
        ic_lower, ic_upper = ic9833_alto_bajo
    elif idx == 1:
        ic_lower, ic_upper = ic9833_alto_medio
    else:
        ic_lower, ic_upper = ic9833_medio_bajo
    
    ax.axvline(ic_lower, color='gray', linestyle=':', linewidth=2)
    ax.axvline(ic_upper, color='gray', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Diferencia en avg_state2')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Diferencias: {nombre}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribuciones Bootstrap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_histogramas_bootstrap.png', dpi=300, bbox_inches='tight')
print("Histogramas guardados en: 02_histogramas_bootstrap.png")

# Boxplots comparativos
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Medias por grupo
ax = axes[0]
data_boxplot_medias = [bootstrap_medias_baja, bootstrap_medias_media, bootstrap_medias_alta]
bp1 = ax.boxplot(data_boxplot_medias, tick_labels=['Baja', 'Media', 'Alta'], patch_artist=True)
for patch, color in zip(bp1['boxes'], ['blue', 'orange', 'green']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('avg_state2', fontsize=12)
ax.set_title('Distribuciones Bootstrap de Medias por Grupo', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Diferencias
ax = axes[1]
data_boxplot_diffs = [bootstrap_diffs_alto_bajo, bootstrap_diffs_alto_medio, bootstrap_diffs_medio_bajo]
bp2 = ax.boxplot(data_boxplot_diffs, tick_labels=['Alta-Baja', 'Alta-Media', 'Media-Baja'], 
                 patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('purple')
    patch.set_alpha(0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=2, label='H0: diff = 0')
ax.set_ylabel('Diferencia en avg_state2', fontsize=12)
ax.set_title('Distribuciones Bootstrap de Diferencias', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Comparacion de Distribuciones Bootstrap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_boxplots_bootstrap.png', dpi=300, bbox_inches='tight')
print("Boxplots guardados en: 02_boxplots_bootstrap.png")

print()
