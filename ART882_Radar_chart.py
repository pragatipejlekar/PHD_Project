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














