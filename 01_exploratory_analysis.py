# Script 01: Analisis Exploratorio Detallado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu

# Configuracion
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("Script 01: Analisis Exploratorio Detallado")
print("="*70)
print()

# Cargar datos
print("Cargando datos...")
df = pd.read_csv('data_processed_with_categories.csv')
print(f"Datos cargados: {len(df)} observaciones")
print()

print("1. Estadisticas Descriptivas por Grupo")
print("-"*70)

# Crear categorias por terciles si no existe
if 'categoria' not in df.columns:
    print("Creando categorias por terciles de df_f0...")
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
    print(f"  Tercil 33%: {tercil_33:.4f}")
    print(f"  Tercil 67%: {tercil_67:.4f}")
    print()

# Estadisticas por grupo
print("Estadisticas de avg_state2 por categoria:")
estadisticas = df.groupby('categoria')['avg_state2'].agg([
    ('n', 'count'),
    ('media', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('Q1', lambda x: x.quantile(0.25)),
    ('mediana', 'median'),
    ('Q3', lambda x: x.quantile(0.75)),
    ('max', 'max')
]).round(4)

# Ordenar por categoria
estadisticas = estadisticas.reindex(['Baja', 'Media', 'Alta'])
print(estadisticas)
print()

# Guardar estadisticas
with open('01_output_estadisticas_grupos.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("Estadisticas Descriptivas por Grupo\n")
    f.write("="*70 + "\n\n")
    f.write(estadisticas.to_string())
    f.write("\n\n")
    f.write("Definicion de categorias:\n")
    f.write("- Baja: df_f0 < {:.4f}\n".format(df[df['categoria']=='Baja']['df_f0'].max()))
    f.write("- Media: df_f0 entre terciles\n")
    f.write("- Alta: df_f0 > {:.4f}\n".format(df[df['categoria']=='Alta']['df_f0'].min()))

print("Estadisticas guardadas en: 01_output_estadisticas_grupos.txt")
print()

print("2. Generando Visualizaciones...")
print("-"*70)

# Crear figura
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Boxplots por grupo
ax1 = fig.add_subplot(gs[0, :])
orden = ['Baja', 'Media', 'Alta']
df_sorted = df.copy()
df_sorted['categoria'] = pd.Categorical(df_sorted['categoria'], categories=orden, ordered=True)
sns.boxplot(data=df_sorted, x='categoria', y='avg_state2', ax=ax1, palette='Set2')
sns.stripplot(data=df_sorted.sample(min(500, len(df))), x='categoria', y='avg_state2', 
             ax=ax1, color='black', alpha=0.2, size=2)
ax1.set_title('Distribucion de avg_state2 por Categoria', fontsize=14, fontweight='bold')
ax1.set_xlabel('Categoria de Actividad Lisosoma', fontsize=12)
ax1.set_ylabel('avg_state2 (probabilidad RE estado alto)', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

# Anadir medias
for i, cat in enumerate(orden):
    media = df[df['categoria']==cat]['avg_state2'].mean()
    ax1.plot(i, media, 'r*', markersize=15, label='Media' if i == 0 else '')
ax1.legend()

# Violin plots
ax2 = fig.add_subplot(gs[1, 0])
sns.violinplot(data=df_sorted, x='categoria', y='avg_state2', ax=ax2, palette='Set2')
ax2.set_title('Violin Plot por Categoria', fontsize=12, fontweight='bold')
ax2.set_xlabel('')
ax2.set_ylabel('avg_state2')
ax2.grid(True, alpha=0.3, axis='y')

# Histogramas superpuestos
ax3 = fig.add_subplot(gs[1, 1:])
for cat, color in zip(orden, ['blue', 'orange', 'green']):
    datos_cat = df[df['categoria']==cat]['avg_state2']
    ax3.hist(datos_cat, bins=30, alpha=0.5, label=cat, color=color, density=True)
ax3.set_xlabel('avg_state2')
ax3.set_ylabel('Densidad')
ax3.set_title('Distribuciones Superpuestas', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Scatter plot con categorias
ax4 = fig.add_subplot(gs[2, :])
df_sample = df.sample(min(3000, len(df)), random_state=42)
colores_scatter = {'Baja': 'blue', 'Media': 'orange', 'Alta': 'green'}
for cat in orden:
    df_cat = df_sample[df_sample['categoria']==cat]
    ax4.scatter(df_cat['df_f0'], df_cat['avg_state2'], 
               alpha=0.4, s=20, label=cat, color=colores_scatter[cat])

# Lineas de tendencia por grupo
for cat, color in zip(orden, ['blue', 'orange', 'green']):
    df_cat = df[df['categoria']==cat]
    z = np.polyfit(df_cat['df_f0'], df_cat['avg_state2'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_cat['df_f0'].min(), df_cat['df_f0'].max(), 100)
    ax4.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.8)

ax4.set_xlabel('df_f0 (actividad lisosoma)', fontsize=12)
ax4.set_ylabel('avg_state2', fontsize=12)
ax4.set_title('Relacion df_f0 vs avg_state2 por Categoria', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('ANALISIS EXPLORATORIO DETALLADO', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('01_exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("Visualizaciones guardadas en: 01_exploratory_analysis.png")
print()
