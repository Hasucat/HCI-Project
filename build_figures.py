"""Reproduce report figures/tables from the supplied per-run result CSVs.

No simulation is run here. The checks establish internal CSV consistency;
they do not validate the underlying SUMO trajectories or metric definitions.
"""
from pathlib import Path
import csv
import json
import math
import statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'evidence'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
CTL = ['fixed_time', 'efficiency', 'equal_bargaining', 'efpb']
LABEL = {'fixed_time':'Fixed time', 'efficiency':'Efficiency',
         'equal_bargaining':'Equal bargaining', 'efpb':'EFPB'}
COLOR = dict(zip(CTL, ['#64748b', '#d18419', '#9274aa', '#087f8c']))
MARKER = dict(zip(CTL, ['o', 's', 'D', '^']))

def read(name):
    with (DATA / name).open(newline='') as f:
        return list(csv.DictReader(f))

runs = read('runs_sumo.csv')
cells = read('cells_sumo.csv')
overall = read('overall_sumo.csv')
paired = read('paired_sumo.csv')
assert len(runs) == 520
assert len({(r['scenario'], r['controller'], r['seed']) for r in runs}) == 520
assert {int(r['seed']) for r in runs} == set(range(4001,4011))
by = defaultdict(list)
for r in runs:
    by[(r['scenario'],r['controller'])].append(r)
assert len(by) == 52 and all(len(v) == 10 for v in by.values())
metrics = ['person_delay_s','max_norm_wait','ped_mean_wait_s','veh_mean_wait_s',
           'collisions','emergency_stops','ped_unserved','veh_unfinished']
max_error = 0.0
for c in cells:
    for key in metrics:
        value = st.mean(float(r[key]) for r in by[(c['scenario'],c['controller'])])
        max_error = max(max_error, abs(value-float(c[key])))
for row in overall:
    chosen = [c for c in cells if c['controller']==row['controller'] and
              (row['group']=='all' or c['group']==row['group'])]
    for key in metrics:
        max_error = max(max_error, abs(st.mean(float(c[key]) for c in chosen)-float(row[key])))
for row in paired:
    other = row['comparison'].split(' - ')[1]
    selected = [r for r in runs if row['group']=='all' or r['group']==row['group']]
    a={(r['scenario'],r['seed']):float(r[row['metric']]) for r in selected if r['controller']=='efpb'}
    b={(r['scenario'],r['seed']):float(r[row['metric']]) for r in selected if r['controller']==other}
    assert a.keys()==b.keys()
    diffs=[a[k]-b[k] for k in a]
    assert len(diffs)==int(row['pairs'])
    max_error=max(max_error,abs(st.mean(diffs)-float(row['mean_diff'])),
                  abs(st.stdev(diffs)/math.sqrt(len(diffs))-float(row['se'])))
assert max_error < 1e-9, max_error
audit={'sumo_runs':len(runs),'cells':len(by),'seeds_per_cell':10,
       'max_numeric_difference':max_error,
       'collisions_recorded':sum(float(r['collisions']) for r in runs),
       'emergency_stops_recorded':sum(float(r['emergency_stops']) for r in runs),
       'scope':'CSV arithmetic verified; raw trajectories and per-run metadata were not supplied.'}
(DATA/'consistency_check.json').write_text(json.dumps(audit,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
 'axes.spines.top':False,'axes.spines.right':False,
 'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':7,
 'pdf.fonttype':42,'savefig.bbox':'tight'})
lookup={(c['scenario'],c['controller']):c for c in cells}
nominal=['C_'+a+b for a in 'LMH' for b in 'LMH']
fig,axes=plt.subplots(2,1,figsize=(3.35,3.65),layout='constrained')
for ax,key,title in zip(axes,['person_delay_s','max_norm_wait'],
 ['(a) Recorded person-delay proxy (s)','(b) Maximum-wait ratio']):
    for j,ctl in enumerate(CTL):
        # Demand cells are categories, not a continuous time series.
        ax.plot(np.arange(9)+(j-1.5)*.13,
                [float(lookup[s,ctl][key]) for s in nominal],
                marker=MARKER[ctl],ms=3.5,ls='none',
                color=COLOR[ctl],label=LABEL[ctl])
    ax.set_xticks(range(9),[s[2:] for s in nominal])
    ax.tick_params(axis='x',labelsize=7)
    ax.set_title(title,loc='left',fontweight='bold')
    ax.grid(axis='y',alpha=.18)
    ax.set_ylim(bottom=0)
axes[1].set_xlabel('Demand cell (pedestrian / vehicle)')
axes[1].axhline(1,color='#343434',ls='--',lw=.9)
axes[1].set_ylim(0,1.08)
axes[1].text(8,.96,'100 s policy limit',ha='right',va='top',fontsize=7)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
fig.savefig(FIG/'nominal_results.pdf');plt.close(fig)

stress=['T_SS','T_HS','T_SH','T_OO']
fig,axes=plt.subplots(2,1,figsize=(3.35,3.65),layout='constrained')
x=np.arange(4)
for ax,key,title in zip(axes,['person_delay_s','max_norm_wait'],
 ['(a) Person-delay proxy (s; log scale)','(b) Maximum-wait ratio (log scale)']):
    for j,ctl in enumerate(CTL):
        vals=[float(lookup[s,ctl][key]) for s in stress]
        # Position, rather than bar length, encodes values on the log axis.
        ax.plot(x+(j-1.5)*.14,vals,ls='none',marker=MARKER[ctl],
                ms=4,color=COLOR[ctl],label=LABEL[ctl])
    ax.set_yscale('log')
    ax.set_xticks(x,[s[2:] for s in stress])
    ax.set_title(title,loc='left',fontweight='bold')
    ax.grid(axis='y',alpha=.18)
axes[1].set_xlabel('Demand cell (pedestrian / vehicle)')
axes[0].set_ylim(1,1500)
axes[1].set_ylim(.1,100)
axes[1].axhline(1,color='#343434',ls='--',lw=.9)
fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
fig.savefig(FIG/'stress_results.pdf');plt.close(fig)

# A column-width flow keeps the diagram beside its system overview.
# The optional branch is not an assertion that an interface was evaluated.
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(3.35,2.5))
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
boxes=[(.83,'Configuration: scenario, controller, seed'),
       (.63,'SUMO vehicles + virtual pedestrian queue'),
       (.43,'Controller selection + signal transitions'),
       (.23,'Recorded metrics + descriptive analysis')]
for y0,txt in boxes:
    ax.add_patch(FancyBboxPatch((.035,y0),.90,.13,boxstyle='round,pad=0.008',
                facecolor='#edf4f6',edgecolor='#50717a',lw=.9))
    ax.text(.485,y0+.065,txt,ha='center',va='center',fontsize=7.3)
for y0 in [.82,.62,.42]:
    ax.annotate('',xy=(.485,y0-.055),xytext=(.485,y0),
                arrowprops={'arrowstyle':'->','lw':.9})
ax.plot([.945,.987,.987,.945],[.495,.495,.09,.09],
        ls='--',lw=.8,color='#6b7280')
ax.text(.48,.09,'Optional: traces and template explanations',
        ha='center',va='center',fontsize=7,color='#425466')
ax.text(.48,.01,'Not evaluated; participant interface remains planned',
        ha='center',fontsize=6.5,color='#6b7280')
fig.savefig(FIG/'pipeline.pdf');plt.close(fig)
print(json.dumps(audit,indent=2))
