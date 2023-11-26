"""
Compare wealth indices values based on the DHS Wealth Index and International Wealth Index (IWI).
"""

from pathlib import Path

import pandas as pd
from scipy.stats import linregress

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


base_path = Path("/home/userx/Desktop/accessible-poverty-estimates/equitable-ai")

input_path = base_path / 'iwi_comparison' / 'ghana_2014_iwi.csv'

output_dir = base_path / 'iwi_comparison'
output_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(input_path)


# -----------------------------------------------------------------------------
# prepare data and calculate dhs wi and iwi quintiles

all_t_df = df[['hhid', 'gender', 'dhs_wi', 'iwi']].copy()

all_t_df['norm_dhs_wi'] = (all_t_df['dhs_wi']-all_t_df['dhs_wi'].min()) / (all_t_df['dhs_wi'].max()-all_t_df['dhs_wi'].min())
all_t_df['norm_iwi'] = (all_t_df['iwi']-all_t_df['iwi'].min()) / (all_t_df['iwi'].max()-all_t_df['iwi'].min())


all_t_df['dhswi_quintile'] = pd.qcut(all_t_df['dhs_wi'], 5, labels=False)
all_t_df['iwi_quintile'] = pd.qcut(all_t_df['iwi'], 5, labels=False)

male_t_df = all_t_df.loc[all_t_df.gender == 'male'].copy()
female_t_df = all_t_df.loc[all_t_df.gender == 'female'].copy()


# -----------------
# plot scatter of male/female dhs wi vs iwi with trend lines

mslope, mintercept, mr_value, mp_value, mstd_err = linregress(male_t_df['norm_dhs_wi'], male_t_df['norm_iwi'])
fslope, fintercept, fr_value, fp_value, fstd_err = linregress(female_t_df['norm_dhs_wi'], female_t_df['norm_iwi'])


ax1 = male_t_df.plot.scatter(x='norm_dhs_wi', y='norm_iwi', c='blue', alpha=0.8)
female_t_df.plot.scatter(ax=ax1, x='norm_dhs_wi', y='norm_iwi', c='red', alpha=0.8)
# ax1.plot(male_t_df['norm_dhs_wi'], mintercept + mslope*male_t_df['norm_dhs_wi'], 'r', label='male trend', color='darkblue', linestyle='--')
ax1.plot(female_t_df['norm_dhs_wi'], fintercept + fslope*female_t_df['norm_dhs_wi'], 'r', label='female trend', color='orange', linestyle='--')
ax1.plot([0,1], [0,1], color='black', linestyle='--')

plt.ylabel('Normalized IWI')
plt.xlabel('Normalized DHS WI')
plt.legend(['male', 'female'], loc='upper left')
plot_path = output_dir / "comparison_scatter.png"
plt.savefig(plot_path)
plt.close()


# -----------------
# create transition matrix of dhs wi to iwi


all_t_df['difference'] = all_t_df['iwi_quintile'] - all_t_df['dhswi_quintile']
male_t_df['difference'] = male_t_df['iwi_quintile'] - male_t_df['dhswi_quintile']
female_t_df['difference'] = female_t_df['iwi_quintile'] - female_t_df['dhswi_quintile']

all_tm_df = pd.crosstab(all_t_df['dhswi_quintile'], all_t_df['iwi_quintile'], normalize='index')
male_tm_df = pd.crosstab(male_t_df['dhswi_quintile'], male_t_df['iwi_quintile'], normalize='index')
female_tm_df = pd.crosstab(female_t_df['dhswi_quintile'], female_t_df['iwi_quintile'], normalize='index')


all_tm_df.to_csv(output_dir / 'all_transition_matrix.csv', index=True)
male_tm_df.to_csv(output_dir / 'male_transition_matrix.csv', index=True)
female_tm_df.to_csv(output_dir / 'female_transition_matrix.csv', index=True)


def get_transition_matrix_stats(df):
    df = df.copy()
    sdf = df.difference.value_counts(normalize=True).reset_index().sort_values('index').rename(columns={'index': 'difference', 'difference': 'percent'})
    s_eq = sdf.loc[sdf.difference == 0, 'percent'].sum()
    s_gt = sdf.loc[sdf.difference > 0, 'percent'].sum()
    s_lt = sdf.loc[sdf.difference < 0, 'percent'].sum()
    return (s_lt, s_eq, s_gt), sdf

all_shift_stats, all_shift_df = get_transition_matrix_stats(all_t_df)
male_shift_stats, male_shift_df = get_transition_matrix_stats(male_t_df)
female_shift_stats, female_shift_df = get_transition_matrix_stats(female_t_df)

all_shift_df.to_csv(output_dir / 'all_shift_df.csv', index=False)
male_shift_df.to_csv(output_dir / 'male_shift_df.csv', index=False)
female_shift_df.to_csv(output_dir / 'female_shift_df.csv', index=False)

shift_stats_df = pd.DataFrame([all_shift_stats, male_shift_stats, female_shift_stats], columns=["poorer_quintile", "same_quintile", "wealthier_quintile"])
shift_stats_df["classification"] = ["all", "male_hoh", "female_hoh"]
shift_stats_df.to_csv(output_dir / 'shift_stats.csv', index=False)
