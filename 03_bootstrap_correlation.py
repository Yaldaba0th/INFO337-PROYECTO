# Script 03: Bootstrap Correlacion

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, ttest_1samp, shapiro
import warnings
warnings.filterwarnings('ignore')

# Configuracion
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("="*70)
print("Script 03: Bootstrap Correlación")
print("="*70)
print()

print("1. Calculando Correlación...")
print("-"*70)

df = pd.read_csv('data_processed_with_categories.csv')
lisosomas = df['particle'].unique()
n_lisosomas = len(lisosomas)

print(f"Datos cargados: {len(df)} observaciones")
print(f"Numero de lisosomas: {n_lisosomas}")
print()

# Correlacion global
rho_obs, p_obs = spearmanr(df['df_f0'], df['avg_state2'])

print("Correlación Global:")
print(f"  Spearman rho = {rho_obs:.4f}")
print(f"  p-valor = {p_obs:.4e}")
print()

print("2. Correlaciones Individuales Lisosoma...")
print("-"*70)

correlaciones_individuales = []

for particle_id in lisosomas:
    df_particle = df[df['particle'] == particle_id]
    
    if len(df_particle) > 10:
        rho_indiv, p_indiv = spearmanr(df_particle['df_f0'], 
                                        df_particle['avg_state2'])
    else:
        rho_indiv = np.nan
        p_indiv = np.nan
    
    correlaciones_individuales.append({
        'Lisosoma': particle_id,
        'n_frames': len(df_particle),
        'rho': rho_indiv,
        'p_valor': p_indiv
    })

correlaciones_df = pd.DataFrame(correlaciones_individuales)

# Estadisticas de correlaciones individuales
rhos_validos = correlaciones_df['rho'].dropna()

print("Estadisticas de correlaciones individuales:")
print(f"  Media: {rhos_validos.mean():.4f}")
print(f"  Mediana: {rhos_validos.median():.4f}")
print(f"  Desv. estandar: {rhos_validos.std():.4f}")
print(f"  Minimo: {rhos_validos.min():.4f}")
print(f"  Maximo: {rhos_validos.max():.4f}")
print()

# Contar cuantos son positivos
n_positivos = np.sum(rhos_validos > 0)
print(f"Lisosomas con correlacion positiva: {n_positivos}/{len(rhos_validos)}")
print()

# Guardar correlaciones individuales
correlaciones_df.to_csv('03_correlaciones_individuales.csv', index=False)
print("Correlaciones individuales guardadas en: 03_correlaciones_individuales.csv")
print()

print("3. Ejecutando Bootstrap por Bloques...")
print("-"*70)

n_bootstrap = 10000
print(f"Iteraciones: {n_bootstrap}")
print(f"Remuestreando {n_lisosomas} lisosomas con reemplazo...")
print()

bootstrap_rhos = []

print("Progreso: ", end='', flush=True)
for i in range(n_bootstrap):
    if i % 1000 == 0:
        print(f"{i}...", end='', flush=True)
    
    # Remuestrear lisosomas
    lisosomas_boot = np.random.choice(lisosomas, size=n_lisosomas, replace=True)
    
    # Construir dataset bootstrap
    df_boot = pd.concat([df[df['particle']==lid] for lid in lisosomas_boot],
                        ignore_index=True)
    
    # Calcular correlacion
    rho_boot, _ = spearmanr(df_boot['df_f0'], df_boot['avg_state2'])
    bootstrap_rhos.append(rho_boot)

print("Completado")
print()

bootstrap_rhos = np.array(bootstrap_rhos)

print("4. Calculando Intervalos de Confianza...")
print("-"*70)

# IC 95% y 99%
ic95 = np.percentile(bootstrap_rhos, [2.5, 97.5])
ic99 = np.percentile(bootstrap_rhos, [0.5, 99.5])

print(f"IC 95%: [{ic95[0]:.4f}, {ic95[1]:.4f}]")
print(f"IC 99%: [{ic99[0]:.4f}, {ic99[1]:.4f}]")
print()

# P-valor bootstrap
p_bootstrap = 2 * min(np.sum(bootstrap_rhos <= 0), np.sum(bootstrap_rhos >= 0)) / n_bootstrap

if p_bootstrap == 0:
    p_boot_str = f"< {1/n_bootstrap:.4f}"
else:
    p_boot_str = f"{p_bootstrap:.4f}"

print(f"P-valor bootstrap: {p_boot_str}")
print()

# Clasificacion segun Schober et al. (2018)
rho_medio = np.mean(bootstrap_rhos)
if abs(rho_medio) < 0.10:
    clasificacion = "Negligible"
elif abs(rho_medio) < 0.38:
    clasificacion = "Debil"
elif abs(rho_medio) < 0.68:
    clasificacion = "Moderada"
elif abs(rho_medio) < 0.89:
    clasificacion = "Fuerte"
else:
    clasificacion = "Muy fuerte"

print(f"Clasificacion (Schober et al. 2018): {clasificacion}")
print()

print("5. Verificando Normalidad (Shapiro-Wilk)...")
print("-"*70)

sample_size = min(5000, n_bootstrap)
sample_shapiro = np.random.choice(bootstrap_rhos, size=sample_size, replace=False)

W, p_shapiro = shapiro(sample_shapiro)

print(f"W-statistic: {W:.4f}")
print(f"p-valor: {p_shapiro:.4e}")
print()

if W >= 0.95:
    print("La distribucion bootstrap es aproximadamente normal (W >= 0.95)")
else:
    print("Desviacion notable de normalidad")
print()

print("6. T-test...")
print("-"*70)

print("T-test de una muestra (H0: rho = 0):")

media_boot = np.mean(bootstrap_rhos)
se_boot = np.std(bootstrap_rhos, ddof=1)
t_stat, p_ttest = ttest_1samp(bootstrap_rhos, 0)

print(f"  Media bootstrap: {media_boot:.4f}")
print(f"  SE bootstrap: {se_boot:.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-valor: {p_ttest:.4e}")
print()

if p_ttest < 0.05:
    print("DECISION: Rechazamos H0 (p < 0.05)")
    print("  La correlacion es significativamente distinta de cero")
else:
    print("DECISION: No rechazamos H0 (p >= 0.05)")
    print("  No hay evidencia de correlacion")
print()


print("7. Guardando Resultados...")
print("-"*70)

with open('03_output_bootstrap_correlacion.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("BOOTSTRAP CORRELACION DE SPEARMAN - RESULTADOS COMPLETOS\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. PARAMETROS DEL ANALISIS\n")
    f.write("-"*70 + "\n")
    f.write(f"  Iteraciones bootstrap: {n_bootstrap}\n")
    f.write(f"  Unidades remuestreadas: {n_lisosomas} lisosomas\n")
    f.write(f"  Metodo: Bootstrap por bloques\n\n")
    
    f.write("2. CORRELACION OBSERVADA\n")
    f.write("-"*70 + "\n")
    f.write(f"  Spearman rho = {rho_obs:.4f}\n")
    f.write(f"  p-valor = {p_obs:.4e}\n\n")
    
    f.write("3. CORRELACIONES INDIVIDUALES POR LISOSOMA\n")
    f.write("-"*70 + "\n")
    f.write(f"  Media: {rhos_validos.mean():.4f}\n")
    f.write(f"  Mediana: {rhos_validos.median():.4f}\n")
    f.write(f"  Desv. std: {rhos_validos.std():.4f}\n")
    f.write(f"  Rango: [{rhos_validos.min():.4f}, {rhos_validos.max():.4f}]\n")
    f.write(f"  Positivos: {n_positivos}/{len(rhos_validos)} ({100*n_positivos/len(rhos_validos):.1f}%)\n\n")
    
    f.write("4. RESULTADOS BOOTSTRAP\n")
    f.write("-"*70 + "\n")
    f.write(f"  Media bootstrap: {media_boot:.4f}\n")
    f.write(f"  SE bootstrap: {se_boot:.4f}\n")
    f.write(f"  IC 95%: [{ic95[0]:.4f}, {ic95[1]:.4f}]\n")
    f.write(f"  IC 99%: [{ic99[0]:.4f}, {ic99[1]:.4f}]\n")
    f.write(f"  Clasificacion: {clasificacion}\n\n")
    
    f.write("5. TESTS POST-BOOTSTRAP\n")
    f.write("-"*70 + "\n")
    f.write(f"  Shapiro-Wilk: W = {W:.4f}, p = {p_shapiro:.4e}\n")
    f.write(f"  T-test: t = {t_stat:.4f}, p = {p_ttest:.4e}\n")
    f.write(f"  Decision: {'Rechaza H0' if p_ttest < 0.05 else 'No rechaza H0'}\n\n")

print("Resultados guardados en: 03_output_bootstrap_correlacion.txt")
print()

print("8. Generando Visualizaciones...")
print("-"*70)

# Analisis de normalidad y distribucion bootstrap
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograma distribucion bootstrap
ax = axes[0, 0]
ax.hist(bootstrap_rhos, bins=50, alpha=0.7, color='green', edgecolor='black', density=True)
ax.axvline(rho_obs, color='red', linewidth=3, label=f'Observado (rho={rho_obs:.3f})')
ax.axvline(ic95[0], color='orange', linestyle='--', linewidth=2, label='IC 95%')
ax.axvline(ic95[1], color='orange', linestyle='--', linewidth=2)
ax.axvline(0, color='black', linestyle=':', linewidth=2, alpha=0.5, label='H0: rho=0')
ax.set_xlabel('Correlacion de Spearman (rho)', fontsize=12)
ax.set_ylabel('Densidad', fontsize=12)
ax.set_title('Distribucion Bootstrap de Correlaciones', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Q-Q plot de normalidad
ax = axes[0, 1]
stats.probplot(bootstrap_rhos, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Normalidad\nDistribucion Bootstrap', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Histograma de correlaciones individuales
ax = axes[1, 0]
ax.hist(rhos_validos, bins=15, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(rhos_validos.mean(), color='red', linewidth=2, 
          label=f'Media={rhos_validos.mean():.3f}')
ax.axvline(rhos_validos.median(), color='orange', linewidth=2, linestyle='--',
          label=f'Mediana={rhos_validos.median():.3f}')
ax.axvline(rho_obs, color='blue', linewidth=2, linestyle=':',
          label=f'Global={rho_obs:.3f}')
ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Correlacion de Spearman (rho)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_title('Correlaciones por Lisosoma', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Boxplot de correlaciones individuales
ax = axes[1, 1]
bp = ax.boxplot(rhos_validos, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)
ax.axhline(rho_obs, color='red', linestyle='--', linewidth=2, label=f'Global: {rho_obs:.3f}')
ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5, label='rho = 0')
ax.set_ylabel('Correlacion (rho)', fontsize=12)
ax.set_title('Distribucion de Correlaciones Individuales', fontsize=13, fontweight='bold')
ax.set_xticklabels(['Por Lisosoma'])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Anadir texto con estadisticas
textstr = f'n = {len(rhos_validos)}\nPositivos: {n_positivos} ({100*n_positivos/len(rhos_validos):.0f}%)'
ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
       verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('BOOTSTRAP CORRELACION DE SPEARMAN (10,000 iteraciones)',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('03_bootstrap_correlacion.png', dpi=300, bbox_inches='tight')
print("Visualizaciones guardadas en: 03_bootstrap_correlacion.png")
print()
