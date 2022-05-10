import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


colour_scheme = {'Tome et al.': 'red',
'L1 Loss': 'deepskyblue',
'Spatial TFM': 'purple',
'Seq. Latent Model': 'blue',
'Direct 3D reg.': 'magenta',
'Ego-STAN Slice': 'teal',
'Ego-STAN Avg': 'darkgreen',
'Ego-STAN FMT': 'limegreen'}

seq_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_seq_hm_direct_05_07_09_46_26/results_xregopose_seq_hm_direct_05_07_09_46_26')

baseline_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_05_07_10_43_25/results_xregopose_05_07_10_43_25')

seq_avg_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_seq_hm_direct_avg_05_07_15_33_32/results_xregopose_seq_hm_direct_avg_05_07_15_33_32')

seq_slice_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_seq_hm_direct_slice_05_07_11_15_19/results_xregopose_seq_hm_direct_slice_05_07_11_15_19')

seq_spatial_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_global_trans_05_07_10_49_12/results_xregopose_global_trans_05_07_10_49_12')

baseline_l1_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_l1_05_07_10_43_26/results_xregopose_l1_05_07_10_43_26')

baseline_direct_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_direct_05_07_10_43_34/results_xregopose_direct_05_07_10_43_34')

seq_latent_results = pd.read_pickle(r'/home/eddie/waterloo/lightning_logs/3d_plots/xregopose_seq_05_07_11_08_22/results_xregopose_seq_05_07_11_08_22')

# comparison_results = {}
# for key, value in baseline_results.items():
#     if key in seq_results:
#         comparison_results[key] = value['full_mpjpe'] - seq_results[key]['full_mpjpe']
#     else:
#         continue


# sorted_files = [k for k, v in sorted(comparison_results.items(), key=lambda item: item[1])]
# sorted_values = [v for k, v in sorted(comparison_results.items(), key=lambda item: item[1])]


pd_data = []
all_actions = []
for file, action, mpjpe in zip(baseline_results['Filenames'], baseline_results['Actions'], baseline_results['All']):
    pd_data.append([file, action, mpjpe*1000, 'Tome et al.'])
    if action not in all_actions:
        all_actions.append(action)

for file, action, mpjpe in zip(baseline_l1_results['Filenames'], baseline_l1_results['Actions'], baseline_l1_results['All']):
    pd_data.append([file, action, mpjpe*1000, 'L1 Loss'])

for file, action, mpjpe in zip(seq_spatial_results['Filenames'], seq_spatial_results['Actions'], seq_spatial_results['All']):
    pd_data.append([file, action, mpjpe*1000, 'Spatial TFM'])

for file, action, mpjpe in zip(seq_latent_results['Filenames'], seq_latent_results['Actions'], seq_latent_results['All']):
    pd_data.append([file, action, mpjpe*1000, 'Seq. Latent Model'])

for file, action, mpjpe in zip(baseline_direct_results['Filenames'], baseline_direct_results['Actions'], baseline_direct_results['All']):
    pd_data.append([file, action, mpjpe*1000, 'Direct 3D reg.'])

for file, action, mpjpe in zip(seq_slice_results['Filenames'], seq_slice_results['Actions'], seq_slice_results['All']):
    pd_data.append([file, action, mpjpe*1000, 'Ego-STAN Slice'])

for file, action, mpjpe in zip(seq_avg_results['Filenames'], seq_avg_results['Actions'], seq_avg_results['All']):
    pd_data.append([file, action, mpjpe*1000, 'Ego-STAN Avg'])

for file, action, mpjpe in zip(seq_results['Filenames'], seq_results['Actions'], seq_results['All']):
    pd_data.append([file, action, mpjpe*1000, 'Ego-STAN FMT'])




df = pd.DataFrame(pd_data, columns=["id", "action", "full_mpjpe", "Model"])


# ax = sns.violinplot(x="action", y="full_mpjpe", data=df, inner="quartile", hue="Model")
# ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
# ax.set_ylabel("MPJPE (mm)")
# ax.set_ylim(0, 100)
# # os.makedirs(output_file, exist_ok=True)
# ax.figure.savefig('/home/eddie/waterloo/lightning_logs/violin_plots/model_comparison_per_action.jpg', bbox_inches = "tight")

baseline_mean = np.mean(df.loc[df['Model'] == 'Tome et al.'])
l1_mean = np.mean(df.loc[df['Model'] == 'L1 Loss'])
direct_mean = np.mean(df.loc[df['Model'] == 'Direct 3D reg.'])
seq_latent_mean = np.mean(df.loc[df['Model'] == 'Seq. Latent Model'])
spatial_mean = np.mean(df.loc[df['Model'] == 'Spatial TFM'])
ego_slice_mean = np.mean(df.loc[df['Model'] == 'Ego-STAN Slice'])
ego_avg_mean = np.mean(df.loc[df['Model'] == 'Ego-STAN Avg'])
ego_mean = np.mean(df.loc[df['Model'] == 'Ego-STAN FMT'])
means = [baseline_mean, l1_mean, spatial_mean, seq_latent_mean, direct_mean, ego_slice_mean, ego_avg_mean, ego_mean]
print(means)

df = pd.DataFrame(pd_data, columns=["id", "action", "full_mpjpe", "Model"])

ax = sns.violinplot(x="Model", y="full_mpjpe", data=df, inner="quartile", palette=colour_scheme.values(), saturation=1)
for ind, l in enumerate(ax.lines):
    
    if ind%3==0:
        l.set_linestyle('-')
    elif ind%3==1:
        l.set_linestyle('--')
    elif ind%3==2:
        l.set_linestyle('dotted')

    if ind<=2:
        l.set_label(f'{25*(ind+1)}% Quartile')

ax.plot(list(range(len(means))), means, linestyle='solid', marker='x', color='black', label='Mean')
# ax.scatter(list(range(len(means))), means, c='black', marker='x')
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
ax.set_ylabel("MPJPE (mm)")
ax.set_ylim(0, 150)
ax.set_xticks([])
ax.set_xlabel('')
ax.legend(loc="upper right")
# os.makedirs(output_file, exist_ok=True)
ax.figure.savefig('/home/eddie/waterloo/lightning_logs/violin_plots/model_comparison.pdf', bbox_inches = "tight", format='pdf')



def bar_plot(ax, data, stds, colors=None, total_width=0.8, single_width=1):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    # Iterate over all data
    for i, model in enumerate(data):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        std_data = stds[i, :]
        # Draw a bar for every value of that type
        for x, y in enumerate(model):
            if x == 0:
                ax.bar(x + x_offset, y, yerr=std_data[x], width=bar_width * single_width, label=list(colour_scheme.keys())[i],
                 color=colors[i % len(colors)], capsize=2, zorder=3)#, hatch=hatches[i], alpha=0.99)
            else:
                ax.bar(x + x_offset, y, yerr=std_data[x], width=bar_width * single_width,
                 color=colors[i % len(colors)], capsize=2, zorder=3)#, hatch=hatches[i], alpha=0.99)





temp_avgs = np.empty([8, 9])
temp_stds = np.empty([8, 9])
all_actions_string = ['Upper \n Stretching', 'Lower \n Stretching', 'Walking', 'Patting', 'Greeting', 'Talking', 'Gesticuling', 'Reacting', 'Gaming']

for i in range(9):
    for j in range(8):
        temp_data = df[(df['Model'] == list(colour_scheme.keys())[j]) & (df['action'] == all_actions[i])]['full_mpjpe']
        averages = np.mean(temp_data)
        stds = np.std(temp_data)
        temp_avgs[j, i] = averages
        temp_stds[j, i] = stds



    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # ax.set_xticks([])
    if i%3 == 0:
        ax.set_ylabel('MPJPE (mm)')

print(all_actions)
fig, ax = plt.subplots(1, 1, figsize=(20,6))
bar_plot(ax, temp_avgs, temp_stds, colors=list(colour_scheme.values()))

# it would of course be better with a nicer handle to the middle-bottom axis object, but since I know it is the second last one in my 3 x 3 grid...
# plt.tight_layout()
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
ax.set_xticklabels(all_actions_string, fontsize=20)
ax.set_ylabel('MPJPE (mm)', fontsize=20)
ax.grid(axis='y', zorder=0)
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
legend = fig.legend(lines, labels, title='Model', bbox_to_anchor=(0.9, -0.05), ncol=4, fontsize=20)
legend.get_title().set_fontsize('20')
fig.savefig('/home/eddie/waterloo/lightning_logs/violin_plots/bar_plot_actions.pdf', bbox_inches='tight', format='pdf')