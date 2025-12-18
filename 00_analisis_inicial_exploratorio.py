# Script 00: Analisis Inicial Exploratorio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, spearmanr, pearsonr
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

# Configuracion de visualizacion
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("Script 00: Analisis Inicial Exploratorio")
print("="*70)
print()

print("1. Cargando Datos...")
print("-"*70)

# Cargar datos
df = pd.read_csv('data_processed_with_categories.csv')

print(f"Datos cargados exitosamente")
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print()

# Informacion basica
print("Variables disponibles:")
print(df.columns.tolist())
print()

print("Primeras filas:")
print(df.head())
print()

# Estadisticas descriptivas
print("Estadisticas descriptivas de variables principales:")
descriptivas = df[['df_f0', 'avg_state0', 'avg_state1', 'avg_state2']].describe()
print(descriptivas)
print()

# Informacion por lisosoma
print("Informacion por lisosoma:")
lisosomas_info = df.groupby('particle').size().describe()
print(f"Numero de lisosomas: {df['particle'].nunique()}")
print(f"Frames por lisosoma (estadisticas):")
print(lisosomas_info)
print()

# Guardar informacion basica
with open('00_output_info_basica.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("Informacion Basica\n")
    f.write("="*70 + "\n\n")
    f.write(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas\n")
    f.write(f"Numero de lisosomas: {df['particle'].nunique()}\n")
    f.write(f"Frames totales: {len(df)}\n\n")
    f.write("Estadisticas descriptivas:\n")
    f.write(descriptivas.to_string())
    f.write("\n\nFrames por lisosoma:\n")
    frames_por_lisosoma = df.groupby('particle').size()
    f.write(frames_por_lisosoma.to_string())

print("Informacion basica guardada en: 00_output_info_basica.txt")
print()

print("2. Tests de Normalidad")
print("-"*70)

variables_test = ['df_f0', 'avg_state0', 'avg_state1', 'avg_state2']
resultados_normalidad = []

for var in variables_test:
    # Shapiro-Wilk test
    if len(df[var]) > 5000:
        muestra = df[var].sample(5000, random_state=42)
        stat, p_valor = shapiro(muestra)
        print(f"{var} (muestra n=5000):")
    else:
        stat, p_valor = shapiro(df[var])
        print(f"{var}:")
    
    print(f"  Shapiro-Wilk statistic: {stat:.4f}")
    print(f"  p-valor: {p_valor:.4e}")
    
    if p_valor < 0.05:
        print(f"  Rechaza normalidad (p < 0.05)")
        normal = "NO"
    else:
        print(f"  No rechaza normalidad (p >= 0.05)")
        normal = "SI"
    print()
    
    resultados_normalidad.append({
        'Variable': var,
        'Shapiro_W': stat,
        'p_valor': p_valor,
        'Normal': normal
    })

# Guardar resultados de normalidad
with open('00_output_normalidad.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("Tests de Normalidad (Shapiro-Wilk)\n")
    f.write("="*70 + "\n\n")
    for res in resultados_normalidad:
        f.write(f"Variable: {res['Variable']}\n")
        f.write(f"  Estadistico W: {res['Shapiro_W']:.4f}\n")
        f.write(f"  p-valor: {res['p_valor']:.4e}\n")
        f.write(f"  Normal: {res['Normal']}\n\n")

print("Tests de normalidad guardados en: 00_output_normalidad.txt")
print()

# Crear Q-Q plots
print("Generando Q-Q plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, var in enumerate(variables_test):
    stats.probplot(df[var], dist="norm", plot=axes[i])
    axes[i].set_title(f'Q-Q Plot: {var}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('00_qqplots.png', dpi=300, bbox_inches='tight')
print("Q-Q plots guardados en: 00_qqplots.png")
print()

print("3. Analisis de Autocorrelación")
print("-"*70)

# Calcular autocorrelacion para cada lisosoma
autocorr_results = []

print("Calculando autocorrelacion lag-1 para cada lisosoma...")
for particle_id in df['particle'].unique():
    df_particle = df[df['particle'] == particle_id].sort_values('frame')
    
    # ACF para df_f0
    if len(df_particle) > 10:
        acf_values = acf(df_particle['df_f0'].values, nlags=1, fft=False)
        autocorr_lag1 = acf_values[1]
    else:
        autocorr_lag1 = np.nan
    
    autocorr_results.append({
        'Lisosoma': particle_id,
        'n_frames': len(df_particle),
        'autocorr_lag1': autocorr_lag1
    })

autocorr_df = pd.DataFrame(autocorr_results)

# Estadisticas de autocorrelacion
print("\nEstadisticas de autocorrelacion lag-1:")
print(f"Media: {autocorr_df['autocorr_lag1'].mean():.4f}")
print(f"Mediana: {autocorr_df['autocorr_lag1'].median():.4f}")
print(f"Desviacion estandar: {autocorr_df['autocorr_lag1'].std():.4f}")
print(f"Minimo: {autocorr_df['autocorr_lag1'].min():.4f}")
print(f"Maximo: {autocorr_df['autocorr_lag1'].max():.4f}")
print()

# Guardar resultados de autocorrelacion
with open('00_output_autocorrelacion.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("Analisis de Autocorrelación\n")
    f.write("="*70 + "\n\n")
    f.write("Autocorrelacion lag-1 por lisosoma:\n")
    f.write(autocorr_df.to_string(index=False))
    f.write("\n\n" + "="*70 + "\n")
    f.write("Estadisticas de Resumen:\n")
    f.write("="*70 + "\n")
    f.write(f"Media: {autocorr_df['autocorr_lag1'].mean():.4f}\n")
    f.write(f"Mediana: {autocorr_df['autocorr_lag1'].median():.4f}\n")
    f.write(f"Desviacion estandar: {autocorr_df['autocorr_lag1'].std():.4f}\n")
    f.write(f"Minimo: {autocorr_df['autocorr_lag1'].min():.4f}\n")
    f.write(f"Maximo: {autocorr_df['autocorr_lag1'].max():.4f}\n\n")

print("Autocorrelacion guardada en: 00_output_autocorrelacion.txt")
print()

# Visualizar ACF para algunos lisosomas
print("Generando graficos ACF para lisosomas ejemplo...")
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

lisosomas_ejemplo = df['particle'].unique()[:9]
for i, particle_id in enumerate(lisosomas_ejemplo):
    df_particle = df[df['particle'] == particle_id].sort_values('frame')
    plot_acf(df_particle['df_f0'], lags=20, ax=axes[i], alpha=0.05)
    axes[i].set_title(f'Lisosoma {particle_id}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('00_acf_plots.png', dpi=300, bbox_inches='tight')
print("Graficos ACF guardados en: 00_acf_plots.png")
print()

print("4. Correlaciones de estados del RE con df_f0")
print("-"*70)

estados = ['avg_state0', 'avg_state1', 'avg_state2']
resultados_correlaciones = []

print("Calculando correlaciones de Spearman:")
for estado in estados:
    rho_spearman, p_spearman = spearmanr(df['df_f0'], df[estado])
    
    if p_spearman < 0.001:
        sig = "***"
    elif p_spearman < 0.01:
        sig = "**"
    elif p_spearman < 0.05:
        sig = "*"
    else:
        sig = "ns"
    
    print(f"{estado}:")
    print(f"  Spearman rho = {rho_spearman:.4f} ({sig})")
    print(f"  p-valor = {p_spearman:.4e}")
    print()
    
    resultados_correlaciones.append({
        'Estado': estado,
        'Spearman_rho': rho_spearman,
        'Spearman_p': p_spearman,
        'Significancia': sig
    })

# Guardar correlaciones
with open('00_output_correlaciones_3estados.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("Correlaciones de estados del RE con df_f0\n")
    f.write("="*70 + "\n\n")
    for res in resultados_correlaciones:
        f.write(f"{res['Estado']}:\n")
        f.write(f"  Spearman rho: {res['Spearman_rho']:.4f}\n")
        f.write(f"  p-valor: {res['Spearman_p']:.4e}\n")
        f.write(f"  Significancia: {res['Significancia']}\n\n")

print("Correlaciones guardadas en: 00_output_correlaciones_3estados.txt")
print()

# Crear scatter plots de los 3 estados
print("Generando scatter plots de los 3 estados...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df_sample = df.sample(3000, random_state=42)

for i, (estado, ax) in enumerate(zip(estados, axes)):
    ax.scatter(df_sample['df_f0'], df_sample[estado], alpha=0.3, s=15)
    
    z = np.polyfit(df['df_f0'], df[estado], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['df_f0'].min(), df['df_f0'].max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.8)
    
    rho = resultados_correlaciones[i]['Spearman_rho']
    sig = resultados_correlaciones[i]['Significancia']
    ax.set_title(f'{estado} vs df_f0\nrho = {rho:.3f} {sig}')
    ax.set_xlabel('df_f0 (actividad lisosoma)')
    ax.set_ylabel(estado)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('00_scatter_3estados.png', dpi=300, bbox_inches='tight')
print("Scatter plots guardados en: 00_scatter_3estados.png")
print()

print("5. Calculos para Bootstrap")
print("-"*70)

# Tamano muestral efectivo aproximado
autocorr_media = autocorr_df['autocorr_lag1'].mean()
n_frames_total = len(df)
n_lisosomas = df['particle'].nunique()
n_efectivo_frames = n_frames_total * (1 - autocorr_media) / (1 + autocorr_media)

print(f"Observaciones totales: {n_frames_total}")
print(f"Autocorrelacion promedio: {autocorr_media:.4f}")
print(f"Tamano muestral efectivo (aprox): {n_efectivo_frames:.0f}")
print()

# Guardar justificacion
with open('00_output_justificacion_bootstrap.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("Calculos para Bootstrap\n")
    f.write("="*70 + "\n\n")
    f.write(f"1. Observaciones totales: {n_frames_total}\n")
    f.write(f"2. Autocorrelacion temporal promedio: {autocorr_media:.4f}\n")
    f.write(f"3. Tamano muestral efectivo (aprox): {n_efectivo_frames:.0f}\n")

print("Justificacion guardada en: 00_output_justificacion_bootstrap.txt")
print()

print("6. Generando Visualizacion...")
print("-"*70)

# Crear figura
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Histogramas de las variables
ax1 = fig.add_subplot(gs[0, :])
df[['df_f0', 'avg_state0', 'avg_state1', 'avg_state2']].hist(bins=50, ax=ax1, 
                                                               layout=(1, 4), 
                                                               figsize=(16, 3))
ax1.set_title('Distribuciones de Variables', fontsize=14, fontweight='bold', pad=20)

# Boxplot de autocorrelacion por lisosoma
ax2 = fig.add_subplot(gs[1, 0])
ax2.boxplot(autocorr_df['autocorr_lag1'].dropna())
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='r=0.3')
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='r=0.5')
ax2.set_ylabel('Autocorrelacion lag-1')
ax2.set_title('Autocorrelacion por Lisosoma')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Correlaciones de los 3 estados
ax3 = fig.add_subplot(gs[1, 1:])
estados_labels = ['State 0', 'State 1', 'State 2']
correlaciones_valores = [r['Spearman_rho'] for r in resultados_correlaciones]
colores = ['red' if x < 0 else 'green' for x in correlaciones_valores]
bars = ax3.bar(estados_labels, correlaciones_valores, color=colores, alpha=0.7)
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.set_ylabel('Correlacion de Spearman (rho)')
ax3.set_title('Correlaciones de Estados del RE con df_f0')
ax3.set_ylim(-0.6, 0.6)
ax3.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, correlaciones_valores)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.05,
             f'{val:.3f}\n{resultados_correlaciones[i]["Significancia"]}',
             ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

# Scatter plot del estado mas correlacionado
ax4 = fig.add_subplot(gs[2, :])
idx_max = np.argmax([abs(r['Spearman_rho']) for r in resultados_correlaciones])
estado_max = estados[idx_max]
df_sample = df.sample(3000, random_state=42)
scatter = ax4.scatter(df_sample['df_f0'], df_sample[estado_max], 
                     c=df_sample['df_f0'], cmap='viridis', 
                     alpha=0.4, s=20, edgecolors='none')
z = np.polyfit(df['df_f0'], df[estado_max], 1)
p = np.poly1d(z)
x_line = np.linspace(df['df_f0'].min(), df['df_f0'].max(), 100)
ax4.plot(x_line, p(x_line), "r-", linewidth=3, alpha=0.8, label='Tendencia lineal')
ax4.set_xlabel('df_f0 (actividad lisosoma)', fontsize=12)
ax4.set_ylabel(estado_max, fontsize=12)
rho_max = resultados_correlaciones[idx_max]['Spearman_rho']
sig_max = resultados_correlaciones[idx_max]['Significancia']
ax4.set_title(f'Estado mas correlacionado: {estado_max} (rho = {rho_max:.3f} {sig_max})', 
             fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='df_f0')

plt.suptitle('ANALISIS EXPLORATORIO INICIAL - RESUMEN', 
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig('00_resumen_visual_completo.png', dpi=300, bbox_inches='tight')
print("Resumen visual guardado en: 00_resumen_visual_completo.png")
print()
