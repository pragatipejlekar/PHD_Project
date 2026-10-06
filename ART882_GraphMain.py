import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches




def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    """Truncate the color map between min_val and max_val."""
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n)))
    return new_cmap

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
methods = ['        Pegasus\n       QSVM', 'BERT\nFNN', 'LSA–SVM            ', 'TextRank             \nRF', '            GloVe–CNN']
cos_sim1 = [0.297, 0.623, 0.781, 0.376, 0.553]
cos_sim2 = [0.158, 0.347, 0.425, 0.819, 0.773]
N = len(methods)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
cos_sim1 += cos_sim1[:1]
cos_sim2 += cos_sim2[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.plot(angles, cos_sim1, linewidth=2.5, color='black', label='Financial summarization')
ax.fill(angles, cos_sim1, color='blue',alpha=0.25)
ax.plot(angles, cos_sim2, linewidth=2.5, color='crimson', label='Gov-report summarization')
ax.fill(angles, cos_sim2,color='orange', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(methods, fontsize=22, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=24, fontweight='bold')
plt.title("Word Error Rate (%)", fontsize=20, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 1.1), fontsize=22)
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig('ART882_Result\\Word Error Rate.png', dpi=300, bbox_inches='tight')
plt.show()



cos_sim1 = [0.31, 0.729, 0.816, 0.448, 0.762]
cos_sim2 = [0.164, 0.336, 0.519, 0.972, 0.831]
N = len(methods)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
cos_sim1 += cos_sim1[:1]
cos_sim2 += cos_sim2[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.plot(angles, cos_sim1, linewidth=2.5, color='black', label='Financial summarization')
ax.fill(angles, cos_sim1, color='blue', alpha=0.25)
ax.plot(angles, cos_sim2, linewidth=2.5, color='crimson', label='Gov-report summarization')
ax.fill(angles, cos_sim2, color='orange', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(methods, fontsize=22, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=24, fontweight='bold')
plt.title("Character Error Rate(%)", fontsize=20, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 1.1), fontsize=22)
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig('ART882_Result\\Character Error Rate.png', dpi=300, bbox_inches='tight')
plt.show()


cos_sim1 = [0.839*100,0.811*100,0.807*100,0.746*100,0.728*100]  ##0.15
cos_sim2 = [0.985*100,0.939*100,0.863*100,0.779*100,0.756*100]
N = len(methods)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
cos_sim1 += cos_sim1[:1]
cos_sim2 += cos_sim2[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.plot(angles, cos_sim1, linewidth=2.5, color='black', label='Financial summarization')
ax.fill(angles, cos_sim1, color='blue', alpha=0.25)
ax.plot(angles, cos_sim2, linewidth=2.5, color='crimson', label='Gov-report summarization')
ax.fill(angles, cos_sim2, color='orange', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(methods, fontsize=22, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=24, fontweight='bold')
plt.title("Normalized Levenshtein Similarity(%)", fontsize=20, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.999), fontsize=22)
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig('ART882_Result\\Normalized Levenshtein Similarity.png', dpi=300, bbox_inches='tight')
plt.show()


cos_sim1 = [0.29,0.62,0.78,0.37,0.55]
cos_sim2 = [0.15,0.34,0.42,0.81,0.77]
N = len(methods)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
cos_sim1 += cos_sim1[:1]
cos_sim2 += cos_sim2[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.plot(angles, cos_sim1, linewidth=2.5, color='black', label='Financial summarization')
ax.fill(angles, cos_sim1, color='blue', alpha=0.25)
ax.plot(angles, cos_sim2, linewidth=2.5, color='crimson', label='Gov-report summarization')
ax.fill(angles, cos_sim2, color='orange', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(methods, fontsize=22, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=24, fontweight='bold')
plt.title("MAPE(%)", fontsize=20, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.999), fontsize=22)
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig('ART882_Result\\MAPE.png', dpi=300, bbox_inches='tight')
plt.show()

acc1 = [0.9083*100,0.914*100,0.905*100,0.858*100,0.834*100]
acc2 = [0.984*100,0.886*100,0.932*100,0.948*100,0.826*100]
N = len(methods)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
acc1 += acc1[:1]
acc2 += acc2[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.plot(angles, acc1, linewidth=2.5, color='black', label='Financial summarization')
ax.fill(angles, acc1, color='blue', alpha=0.25)
ax.plot(angles, acc2, linewidth=2.5, color='crimson', label='Gov-report summarization')
ax.fill(angles, acc2, color='orange', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(methods, fontsize=22, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=24, fontweight='bold')
plt.title("Accuracy(%)", fontsize=20, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.999), fontsize=22)
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig('ART882_Result\\Accuracy.png', dpi=300, bbox_inches='tight')
plt.show()


pre1 = [0.96*100,0.895*100,0.848*100,0.853*100,0.831*100]
pre2 = [0.963*100,0.911*100,0.859*100,0.896*100,0.848*100]
N = len(methods)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
pre1 += pre1[:1]
pre2 += pre2[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.plot(angles, pre1, linewidth=2.5, color='black', label='Financial summarization')
ax.fill(angles, pre1, color='blue', alpha=0.25)
ax.plot(angles, pre2, linewidth=2.5, color='crimson', label='Gov-report summarization')
ax.fill(angles, pre2, color='orange', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(methods, fontsize=22, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=24, fontweight='bold')
plt.title("Precision(%)", fontsize=20, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.999), fontsize=22)
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig('ART882_Result\\Precision.png', dpi=300, bbox_inches='tight')
plt.show()


Recall1 = [0.898*100,0.884*100,0.852*100,0.825*100,0.834*100]  ##0.1
Recall2 = [0.974*100,0.8136*100,0.8325*100,0.8473*100,0.8611*100]
N = len(methods)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
Recall1 += Recall1[:1]
Recall2 += Recall2[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.plot(angles, Recall1, linewidth=2.5, color='black', label='Financial summarization')
ax.fill(angles, Recall1, color='blue', alpha=0.25)
ax.plot(angles, Recall2, linewidth=2.5, color='crimson', label='Gov-report summarization')
ax.fill(angles, Recall2, color='orange', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(methods, fontsize=22, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=24, fontweight='bold')
plt.title("Recall(%)", fontsize=20, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.999), fontsize=22)
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig('ART882_Result\\Recall.png', dpi=300, bbox_inches='tight')
plt.show()


sp1=[(Recall1[0]+pre1[0]-100),(Recall1[1]+pre1[1]-100),(Recall1[2]+pre1[2]-100),(Recall1[3]+pre1[3]-100),(Recall1[4]+pre1[4]-100)]  ##0.1
sp2=[(Recall2[0]+pre2[0]-100),(Recall2[1]+pre2[1]-100),(Recall2[2]+pre2[2]-100),(Recall2[3]+pre2[3]-100),(Recall2[4]+pre2[4]-100)]
N = len(methods)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
sp1 += sp1[:1]
sp2 += sp2[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.plot(angles, sp1, linewidth=2.5, color='black', label='Financial summarization')
ax.fill(angles, sp1, color='blue', alpha=0.25)
ax.plot(angles, sp2, linewidth=2.5, color='crimson', label='Gov-report summarization')
ax.fill(angles, sp2, color='orange', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(methods, fontsize=22, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=24, fontweight='bold')
plt.title("F1score(%)", fontsize=20, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.999), fontsize=22)
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig('ART882_Result\\F1score.png', dpi=300, bbox_inches='tight')
plt.show()
























methods = ['MAM-BERT', 'RoBERTa-SUM', 'BERTSUM', 'BigBird', 'ProphetNet']
cos_sim1 = [0.305*100, 0.4166*100, 0.372*100, 0.448*100, 0.4353*100]
cos_sim2 = [0.167*100, 0.289*100, 0.336*100, 0.397*100, 0.352*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(cos_sim1, cos_sim2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.1, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.1, f"{val2:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("Compression Ratio(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 50)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\Compression Ratio.png', dpi=300, bbox_inches='tight')
plt.show()



cos_sim1 = [0.653, 0.569, 0.491, 0.434, 0.525]
cos_sim2 = [0.636, 0.581, 0.527, 0.406, 0.462]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(cos_sim1, cos_sim2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.001, f"{val1:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.001, f"{val2:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("Lexical Density", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 0.7)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\Lexical Density.png', dpi=300, bbox_inches='tight')
plt.show()




cos_sim1 = [0.346, 0.417, 0.443, 0.566, 0.626]
cos_sim2 = [0.363, 0.428, 0.539, 0.591, 0.618]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(cos_sim1, cos_sim2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.001, f"{val1:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.001, f"{val2:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("Stopword Ratio", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 0.7)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\Stopword Ratio.png', dpi=300, bbox_inches='tight')
plt.show()





cos_sim1 = [0.566, 0.482, 0.424, 0.491, 0.455]
cos_sim2 = [0.646, 0.412, 0.538, 0.469, 0.405]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(cos_sim1, cos_sim2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.001, f"{val1:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.001, f"{val2:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("METEOR Score", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 0.7)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\METEOR Score.png', dpi=300, bbox_inches='tight')
plt.show()




cos_sim1 = [0.529, 0.463, 0.429, 0.468, 0.371]
cos_sim2 = [0.512, 0.431, 0.364, 0.489, 0.343]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(cos_sim1, cos_sim2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.001, f"{val1:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.001, f"{val2:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("Jensen-Shannon Divergence", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 0.6)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\Jensen-Shannon Divergence.png', dpi=300, bbox_inches='tight')
plt.show()




kkkkkk


#######     CLASSIFICATION    ###################

methods = ['MAM-BERT', 'RoBERTa-SUM', 'BERTSUM', 'BigBird', 'ProphetNet']
cos_sim1 = [0.995*100, 0.923*100, 0.895*100, 0.917*100, 0.873*100]
cos_sim2 = [0.996*100, 0.968*100, 0.954*100, 0.943*100, 0.909*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(cos_sim1, cos_sim2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 1, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 1, f"{val2:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("Cosine Similarity(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\Cosine Similarity.png', dpi=300, bbox_inches='tight')
plt.show()





jac1 = [0.7677*100,0.737*100,0.694*100,0.665*100,0.652*100]
jac2 = [0.833*100,0.767*100,0.721*100,0.716*100,0.704 *100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(jac1, jac2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 1, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 1, f"{val2:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("Jaccard Similarity(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 90)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\Jaccard Similarity.png', dpi=300, bbox_inches='tight')
plt.show()


rough1_1 = [0.8295*100,0.817*100,0.782*100,0.725*100,0.793*100]
rough1_2 = [0.9129*100,0.853*100,0.815*100,0.872*100,0.846*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(rough1_1, rough1_2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 1, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 1, f"{val2:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("ROUGE-1(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\ROUGE-1.png', dpi=300, bbox_inches='tight')
plt.show()



rough2_1 = [0.813*100,0.738*100,0.744*100,0.73*100,0.68*100]
rough2_2 = [0.906*100,0.89*100,0.824*100,0.798*100,0.852*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(rough2_1, rough2_2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 1, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 1, f"{val2:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("ROUGE-2(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\ROUGE-2.png', dpi=300, bbox_inches='tight')
plt.show()




roughL_1 = [0.8221*100,0.779*100,0.738*100,0.749*100,0.817*100]
roughL_2 = [0.9129*100,0.848*100,0.871*100,0.825*100,0.796*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(roughL_1, roughL_2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 1, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 1, f"{val2:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("ROUGE-L(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\ROUGE-L.png', dpi=300, bbox_inches='tight')
plt.show()



sem_sim_1 = [0.959*100,0.888*100,0.924*100,0.898*100,0.852*100]
sem_sim_2 = [0.972*100,0.898*100,0.906*100,0.858*100,0.817*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(sem_sim_1, sem_sim_2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 1, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 1, f"{val2:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("Semantic Similarity(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\Semantic Similarity.png', dpi=300, bbox_inches='tight')
plt.show()


sem_sim_1 = [8.24,6.41,6.76,5.59,4.22]
sem_sim_2 = [8.27,7.69,5.93,4.62,6.11]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(sem_sim_1, sem_sim_2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.01, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.01, f"{val2:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("Entropy", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 10)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\Entropy.png', dpi=300, bbox_inches='tight')
plt.show()






sem_sim_1 = [0.93*100,0.874*100,0.852*100,0.787*100,0.816*100]
sem_sim_2 = [0.9332*100,0.85*100,0.91*100,0.883*100,0.823*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(sem_sim_1, sem_sim_2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.01, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.01, f"{val2:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("BERTScore F1Score", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\BERTScore F1.png', dpi=300, bbox_inches='tight')
plt.show()



sem_sim_1 = [0.925*100,0.879*100,0.832*100,0.861*100,0.783*100]
sem_sim_2 = [0.93*100,0.874*100,0.773*100,0.791*100,0.849*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(sem_sim_1, sem_sim_2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.01, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.1, f"{val2:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("BERTScore Precision", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\BERTScore Precision.png', dpi=300, bbox_inches='tight')
plt.show()




sem_sim_1 = [0.936*100,0.929*100,0.847*100,0.783*100,0.813*100]
sem_sim_2 = [0.933*100,0.914*100,0.864*100,0.875*100,0.882*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(sem_sim_1, sem_sim_2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.1, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.1, f"{val2:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("BERTScore Recall", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\BERTScore Recall.png', dpi=300, bbox_inches='tight')
plt.show()





sem_sim_1 = [0.75*100,0.724*100,0.694*100,0.716*100,0.672*100]
sem_sim_2 = [0.833*100,0.767*100,0.758*100,0.746*100,0.70*100]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35  
for i, (val1, val2) in enumerate(zip(sem_sim_1, sem_sim_2)):
    
    x1 = i - bar_width/2
    x2 = i + bar_width/2
    width_bottom = 0.3
    width_top = 0.15
    points1 = np.array([
        [x1 - width_bottom / 2, 0],
        [x1 + width_bottom / 2, 0],
        [x1 + width_top / 2, val1],
        [x1 - width_top / 2, val1]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent1 = [x1 - width_bottom / 2, x1 + width_bottom / 2, 0, val1]
    cmap1 = truncate_colormap(plt.get_cmap('RdPu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent1, origin='lower', cmap=cmap1, alpha=0.8, zorder=1)
    ax.text(x1, val1 + 0.01, f"{val1:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    points2 = np.array([
        [x2 - width_bottom / 2, 0],
        [x2 + width_bottom / 2, 0],
        [x2 + width_top / 2, val2],
        [x2 - width_top / 2, val2]
    ])
    extent2 = [x2 - width_bottom / 2, x2 + width_bottom / 2, 0, val2]
    cmap2 = truncate_colormap(plt.get_cmap('YlGnBu'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent2, origin='lower', cmap=cmap2, alpha=0.8, zorder=1)
    ax.text(x2, val2 + 0.01, f"{val2:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, -0.01, methods[i], ha='center', va='top', fontsize=16, fontname='Times New Roman', color='black')
ax.set_ylabel("Coverage Score(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 90)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
patch1 = mpatches.Patch(color=plt.get_cmap('RdPu')(0.5), label='Financial summarization')
patch2 = mpatches.Patch(color=plt.get_cmap('YlGnBu')(0.5), label='Gov-report summarization')
ax.legend(handles=[patch1, patch2], fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('ART882_Result\\Coverage Score.png', dpi=300, bbox_inches='tight')
plt.show()


















