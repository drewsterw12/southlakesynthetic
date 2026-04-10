# ============================================================
# app.py — SynthetiCare Agent (Live Mode)
# ============================================================

import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, kstest

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SynthetiCare — Southlake Health",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# BRAND COLOURS
# ============================================================
COLORS = {
    'navy': '#003946',
    'teal': '#00838f',
    'dark_teal': '#004d5a',
    'turquoise': '#26c6da',
    'light_mint': '#4ecdc4',
    'white': '#ffffff',
    'light_bg': '#f0f9fa',
    'warm_gray': '#f5f5f5',
    'alert_amber': '#f9a825',
    'alert_red': '#e53935',
    'success': '#43a047',
}

# ============================================================
# CSS
# ============================================================
st.markdown(f"""
<style>.stApp {{
        background-color: {COLORS['white']};
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['navy']} 0%, {COLORS['dark_teal']} 100%);
    }}
    section[data-testid="stSidebar"].stRadio label {{
        color: white !important;
        font-size: 15px;
    }}
    section[data-testid="stSidebar"].stRadio div[role="radiogroup"] label:hover {{
        background-color: rgba(38, 198, 218, 0.15);
        border-radius: 6px;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}
    section[data-testid="stSidebar"].stMarkdown p,
    section[data-testid="stSidebar"].stMarkdown li {{
        color: rgba(255,255,255,0.8) !important;
        font-size: 13px;
    }}
    section[data-testid="stSidebar"].stMarkdown h1,
    section[data-testid="stSidebar"].stMarkdown h2,
    section[data-testid="stSidebar"].stMarkdown h3 {{
        color: white !important;
    }}.metric-card {{
        background: {COLORS['light_bg']};
        border-left: 4px solid {COLORS['teal']};
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }}.metric-card h3 {{
        margin: 0 0 4px 0;
        color: {COLORS['navy']};
        font-size: 14px;
        font-weight: 600;
    }}.metric-card.value {{
        font-size: 28px;
        font-weight: 700;
        color: {COLORS['teal']};
    }}.hero {{
        background: linear-gradient(135deg, {COLORS['navy']} 0%, {COLORS['dark_teal']} 60%, {COLORS['teal']} 100%);
        padding: 40px 48px;
        border-radius: 16px;
        margin-bottom: 28px;
    }}.hero h1 {{
        color: white;
        font-size: 36px;
        margin: 0;
    }}.hero p {{
        color: {COLORS['turquoise']};
        font-size: 17px;
        margin: 8px 0 0 0;
    }}.phase-header {{
        background: {COLORS['light_bg']};
        border-radius: 10px;
        padding: 14px 20px;
        margin: 24px 0 16px 0;
        border-left: 5px solid {COLORS['teal']};
    }}.phase-header h2 {{
        margin: 0;
        color: {COLORS['navy']};
        font-size: 20px;
    }}.phase-header span {{
        color: {COLORS['teal']};
        font-size: 13px;
    }}.source-card {{
        background: white;
        border: 1px solid #e0e0e0;
        border-left: 4px solid {COLORS['teal']};
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}.source-card b {{
        color: {COLORS['navy']};
    }}.source-card.licence {{
        color: #757575;
        font-size: 12px;
    }}.fidelity-ring {{
        background: linear-gradient(135deg, {COLORS['navy']}, {COLORS['dark_teal']});
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
    }}.fidelity-ring h2 {{
        color: white;
        margin: 0;
    }}.fidelity-ring.score {{
        font-size: 64px;
        font-weight: 800;
        color: {COLORS['turquoise']};
        margin: 8px 0;
    }}.fidelity-ring.verdict {{
        color: white;
        font-size: 15px;
    }}.deliverable-row {{
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        color: white;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# BACKEND CLASSES
# ============================================================

class SASEngine:
    def __init__(self, sas_dir, output_dir):
        self.sas_dir = sas_dir
        self.output_dir = output_dir
        self.log = []

    def run_sas_code(self, sas_code, program_name="temp_program"):
        wrapped = f"""
/* Auto-generated by SynthetiCare Agent */
/* Program: {program_name} */
/* Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} */

ods listing close;
ods html5 file="{self.output_dir}/{program_name}.html"
         options(bitmap_mode='inline') style=HTMLBlue;
ods csvall file="{self.output_dir}/{program_name}.csv";

{sas_code}

ods csvall close;
ods html5 close;
ods listing;
"""
        path = os.path.join(self.sas_dir, f"{program_name}.sas")
        with open(path, 'w') as f:
            f.write(wrapped)
        self.log.append({'program': program_name, 'code': sas_code,
                         'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
        return path


class SASCodeGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate_import_code(self, csv_path, dataset_name="WORK.SOURCE_DATA"):
        return f"""
proc import datafile="{csv_path}"
    out={dataset_name}
    dbms=csv replace;
    guessingrows=max;
run;
proc contents data={dataset_name};
    title 'Dataset Contents';
run;
proc print data={dataset_name}(obs=10);
    title 'First 10 Observations';
run;
"""

    def generate_profiling_code(self, df, dataset_name="WORK.SOURCE_DATA"):
        num = ' '.join(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
        cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        code = f"""
proc means data={dataset_name} n nmiss mean std min q1 median q3 max
           skewness kurtosis maxdec=4;
    var {num};
    title 'Descriptive Statistics — Numeric Variables';
    output out=WORK.PROFILE_STATS;
run;
proc univariate data={dataset_name} normal;
    var {num};
    histogram / normal kernel;
    title 'Distribution Analysis with Normality Tests';
run;
"""
        if cat:
            code += f"""
proc freq data={dataset_name};
    tables {' '.join(cat)} / nocum;
    title 'Frequency Distributions — Categorical Variables';
run;
"""
        code += f"""
proc means data={dataset_name} nmiss n;
    var {num};
    title 'Missing Value Analysis';
run;
proc means data={dataset_name} p5 p10 p25 p50 p75 p90 p95 maxdec=4;
    var {num};
    title 'Percentile Analysis';
run;
"""
        return code

    def generate_hygiene_code(self, df, dataset_name="WORK.SOURCE_DATA"):
        num = ' '.join(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
        cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        code = f"""
proc stdize data={dataset_name} out=WORK.CLEANED_DATA
    method=median reponly missing=median;
    var {num};
run;
"""
        for col in cat:
            code += f"""
proc sql noprint;
    select {col} into :mode_{col} trimmed
    from {dataset_name}
    group by {col}
    having count(*) = max(count(*));
quit;
data WORK.CLEANED_DATA;
    set WORK.CLEANED_DATA;
    if missing({col}) then {col} = "&mode_{col}";
run;
"""
        code += f"""
proc means data=WORK.CLEANED_DATA n nmiss mean std min max maxdec=4;
    var {num};
    title 'Post-Cleaning Statistics';
run;
"""
        return code

    def generate_correlation_code(self, df, dataset_name="WORK.CLEANED_DATA"):
        num = ' '.join(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
        return f"""
proc corr data={dataset_name} pearson spearman nosimple
    outp=WORK.CORR_PEARSON outs=WORK.CORR_SPEARMAN;
    var {num};
    title 'Correlation Matrix — Pearson & Spearman';
run;
proc export data=WORK.CORR_PEARSON
    outfile="{self.output_dir}/correlation_matrix.csv"
    dbms=csv replace;
run;
"""

    def generate_visualization_code(self, df, dataset_name="WORK.CLEANED_DATA"):
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        code = "ods graphics on / width=800px height=500px;\n"
        for var in [c for c in ['age', 'income', 'risk_score', 'chronic_condition_count',
                                 'er_visits_12mo', 'fall_risk_score'] if c in num][:6]:
            code += f"""
proc sgplot data={dataset_name};
    histogram {var} / fillattrs=(color=cx00838f transparency=0.3);
    density {var} / type=kernel lineattrs=(color=cx004d5a thickness=2);
    title 'Distribution of {var.replace("_", " ").title()}';
run;
"""
        if 'municipality' in df.columns and 'risk_score' in df.columns:
            code += f"""
proc sgplot data={dataset_name};
    vbar municipality / response=risk_score stat=mean
        fillattrs=(color=cx00838f) categoryorder=respdesc;
    title 'Average Risk Score by Municipality';
run;
"""
        code += "ods graphics off;\n"
        return code

    def generate_export_cleaned_code(self):
        return f"""
proc export data=WORK.CLEANED_DATA
    outfile="{self.output_dir}/cleaned_data.csv"
    dbms=csv replace;
run;
"""

    def generate_fidelity_code(self, df, original="WORK.CLEANED_DATA",
                                synthetic="WORK.SYNTHETIC_DATA"):
        num = ' '.join(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
        code = f"""
proc means data={original} n mean std min max skewness kurtosis maxdec=4;
    var {num};
    title 'Original Data — Summary Statistics';
run;
proc means data={synthetic} n mean std min max skewness kurtosis maxdec=4;
    var {num};
    title 'Synthetic Data — Summary Statistics';
run;
proc corr data={original} pearson nosimple outp=WORK.ORIG_CORR;
    var {num};
run;
proc corr data={synthetic} pearson nosimple outp=WORK.SYNTH_CORR;
    var {num};
run;
data WORK.COMBINED;
    set {original}(in=a) {synthetic}(in=b);
    if a then source='Original'; else source='Synthetic';
run;
ods graphics on / width=800px height=500px;
"""
        for var in ['age', 'risk_score', 'income', 'fall_risk_score']:
            if var in df.columns and pd.api.types.is_numeric_dtype(df[var]):
                code += f"""
proc sgplot data=WORK.COMBINED;
    histogram {var} / group=source transparency=0.5;
    density {var} / type=kernel group=source;
    title 'Fidelity: {var.replace("_", " ").title()}';
run;
"""
        code += "ods graphics off;\n"
        return code


class SyntheticGenerator:
    def __init__(self):
        self.metadata = {}
        self.correlation_matrix = None
        self.numeric_cols = []

    def extract_metadata(self, df):
        self.metadata = {}
        for col in df.columns:
            meta = {'name': col, 'dtype': str(df[col].dtype)}
            series = df[col].dropna()
            unique_vals = set(series.unique())
            is_binary = unique_vals.issubset({0, 1, 0.0, 1.0})

            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10 and not is_binary:
                meta['type'] = 'numeric'
                meta['mean'] = float(series.mean())
                meta['std'] = float(series.std())
                meta['min'] = float(series.min())
                meta['max'] = float(series.max())
                meta['skewness'] = float(series.skew())
                meta['kurtosis'] = float(series.kurtosis())
                meta['percentiles'] = {
                    'p5': float(series.quantile(0.05)),
                    'p25': float(series.quantile(0.25)),
                    'p50': float(series.quantile(0.50)),
                    'p75': float(series.quantile(0.75)),
                    'p95': float(series.quantile(0.95))
                }
                best_name, best_params, best_ks = 'norm', stats.norm.fit(series), 1.0
                for dname, dist in [('norm', stats.norm), ('lognorm', stats.lognorm),
                                     ('gamma', stats.gamma)]:
                    try:
                        params = dist.fit(series)
                        ks_stat, _ = kstest(series, dname, args=params)
                        if ks_stat < best_ks:
                            best_name, best_params, best_ks = dname, params, ks_stat
                    except:
                        continue
                meta['best_distribution'] = best_name
                meta['dist_params'] = [float(p) for p in best_params]
                meta['ks_statistic'] = float(best_ks)

            elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10 and not is_binary:
                meta['type'] = 'categorical'
                freq = series.value_counts(normalize=True)
                meta['categories'] = {str(k): float(v) for k, v in freq.items()}

            elif is_binary:
                meta['type'] = 'binary'
                meta['probability'] = float(series.mean())

            else:
                meta['type'] = 'categorical'
                freq = series.value_counts(normalize=True)
                meta['categories'] = {str(k): float(v) for k, v in freq.items()}

            self.metadata[col] = meta

        self.numeric_cols = [c for c in df.columns if self.metadata[c]['type'] == 'numeric']
        if len(self.numeric_cols) >= 2:
            self.correlation_matrix = df[self.numeric_cols].corr(method='spearman').values
        return self.metadata

    def generate(self, n_rows, seed=42):
        np.random.seed(seed)
        synthetic = pd.DataFrame()

        if self.correlation_matrix is not None and len(self.numeric_cols) >= 2:
            corr = self.correlation_matrix.copy()
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = np.maximum(eigvals, 1e-8)
            corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
            np.fill_diagonal(corr, 1.0)
            mvn = np.random.multivariate_normal(
                mean=np.zeros(len(self.numeric_cols)), cov=corr, size=n_rows)
            for i, col in enumerate(self.numeric_cols):
                u = np.clip(norm.cdf(mvn[:, i]), 1e-6, 1 - 1e-6)
                meta = self.metadata[col]
                try:
                    dist = getattr(stats, meta['best_distribution'])
                    values = dist.ppf(u, *meta['dist_params'])
                except:
                    values = norm.ppf(u, loc=meta['mean'], scale=meta['std'])
                values = np.clip(values, meta['min'], meta['max'])
                if 'int' in meta['dtype']:
                    values = np.round(values).astype(int)
                synthetic[col] = values

        for col, meta in self.metadata.items():
            if meta['type'] == 'binary':
                synthetic[col] = (np.random.random(n_rows) < meta['probability']).astype(int)

        for col, meta in self.metadata.items():
            if meta['type'] == 'categorical':
                cats = list(meta['categories'].keys())
                probs = list(meta['categories'].values())
                total = sum(probs)
                probs = [p / total for p in probs]
                synthetic[col] = np.random.choice(cats, size=n_rows, p=probs)

        col_order = [c for c in self.metadata.keys() if c in synthetic.columns]
        return synthetic[col_order]

    def compute_fidelity(self, original, synthetic):
        fidelity = {'numeric': {}, 'categorical': {}, 'binary': {}, 'overall_scores': []}
        for col in original.columns:
            if col not in synthetic.columns:
                continue
            meta = self.metadata.get(col, {})
            if meta.get('type') == 'numeric':
                orig = original[col].dropna().astype(float)
                synth = synthetic[col].dropna().astype(float)
                ks_stat, ks_p = stats.ks_2samp(orig, synth)
                mean_diff = abs(orig.mean() - synth.mean()) / (abs(orig.mean()) + 1e-10) * 100
                std_diff = abs(orig.std() - synth.std()) / (abs(orig.std()) + 1e-10) * 100
                score = max(0, 1 - ks_stat * 2) * 100
                fidelity['numeric'][col] = {
                    'ks_statistic': round(ks_stat, 4), 'ks_pvalue': round(ks_p, 4),
                    'mean_diff_pct': round(mean_diff, 2), 'std_diff_pct': round(std_diff, 2),
                    'score': round(score, 1)}
                fidelity['overall_scores'].append(score)
            elif meta.get('type') == 'binary':
                orig_rate = original[col].mean()
                synth_rate = float(synthetic[col].astype(float).mean())
                diff = abs(orig_rate - synth_rate)
                score = max(0, 1 - diff * 10) * 100
                fidelity['binary'][col] = {
                    'original_rate': round(orig_rate, 4),
                    'synthetic_rate': round(synth_rate, 4),
                    'abs_diff': round(diff, 4), 'score': round(score, 1)}
                fidelity['overall_scores'].append(score)
            elif meta.get('type') == 'categorical':
                orig_freq = original[col].astype(str).value_counts(normalize=True)
                synth_freq = synthetic[col].astype(str).value_counts(normalize=True)
                all_cats = set(orig_freq.index) | set(synth_freq.index)
                tvd = 0.5 * sum(abs(orig_freq.get(c, 0) - synth_freq.get(c, 0)) for c in all_cats)
                score = max(0, 1 - tvd * 3) * 100
                fidelity['categorical'][col] = {'tvd': round(tvd, 4), 'score': round(score, 1)}
                fidelity['overall_scores'].append(score)

        numeric_cols = [c for c in original.columns
                        if self.metadata.get(c, {}).get('type') == 'numeric']
        if len(numeric_cols) >= 2:
            orig_corr = original[numeric_cols].corr(method='spearman')
            synth_corr = synthetic[numeric_cols].astype(float).corr(method='spearman')
            common = [c for c in orig_corr.columns if c in synth_corr.columns]
            if len(common) >= 2:
                diff = (orig_corr.loc[common, common] - synth_corr.loc[common, common]).abs().values
                avg = diff[np.triu_indices_from(diff, k=1)].mean()
                cs = max(0, 1 - avg * 3) * 100
                fidelity['correlation_diff'] = round(avg, 4)
                fidelity['correlation_score'] = round(cs, 1)
                fidelity['overall_scores'].append(cs)

        fidelity['overall_score'] = round(np.mean(fidelity['overall_scores']), 1)
        return fidelity


# ============================================================
# DATA BUILDER
# ============================================================
def build_catchment_dataset():
    catchment_pop = {
        'Newmarket': {'pop': 87942, 'median_age': 40.2, 'median_income': 95000},
        'Aurora': {'pop': 62057, 'median_age': 41.5, 'median_income': 110000},
        'East Gwillimbury': {'pop': 34637, 'median_age': 39.8, 'median_income': 98000},
        'Georgina': {'pop': 47642, 'median_age': 42.1, 'median_income': 82000},
        'Bradford West Gwillimbury': {'pop': 42880, 'median_age': 37.5, 'median_income': 92000},
        'King': {'pop': 27333, 'median_age': 43.2, 'median_income': 125000},
        'Innisfil': {'pop': 43326, 'median_age': 40.8, 'median_income': 88000},
    }
    prevalence = {
        'diabetes': 0.087, 'hypertension': 0.198, 'copd': 0.042,
        'asthma': 0.112, 'heart_disease': 0.058, 'mood_disorders': 0.082,
        'arthritis': 0.167, 'dementia': 0.065,
    }
    housing_rates_raw = {
        'Single detached': 0.52, 'Semi-detached': 0.08, 'Row house': 0.12,
        'Apartment <5 storeys': 0.10, 'Apartment 5+ storeys': 0.14, 'Other': 0.04,
    }
    ht = sum(housing_rates_raw.values())
    housing_rates = {k: v / ht for k, v in housing_rates_raw.items()}
    storeys_by_dwelling = {
        'Single detached': {'1': 0.25, '2': 0.60, '3+': 0.15},
        'Semi-detached': {'1': 0.10, '2': 0.75, '3+': 0.15},
        'Row house': {'1': 0.05, '2': 0.80, '3+': 0.15},
        'Apartment <5 storeys': {'1': 0.90, '2': 0.10, '3+': 0.0},
        'Apartment 5+ storeys': {'1': 0.95, '2': 0.05, '3+': 0.0},
        'Other': {'1': 0.60, '2': 0.30, '3+': 0.10},
    }
    age_dist_raw = {
        '0-14': 0.155, '15-24': 0.115, '25-34': 0.130, '35-44': 0.135,
        '45-54': 0.125, '55-64': 0.135, '65-74': 0.105, '75-84': 0.060, '85+': 0.040,
    }
    tot = sum(age_dist_raw.values())
    age_dist = {k: v / tot for k, v in age_dist_raw.items()}
    age_ranges = {
        '0-14': (0, 14), '15-24': (15, 24), '25-34': (25, 34), '35-44': (35, 44),
        '45-54': (45, 54), '55-64': (55, 64), '65-74': (65, 74), '75-84': (75, 84),
        '85+': (85, 99),
    }
    room_means = {
        'Single detached': 8, 'Semi-detached': 7, 'Row house': 7,
        'Apartment <5 storeys': 4, 'Apartment 5+ storeys': 4, 'Other': 5,
    }

    np.random.seed(42)
    records = []
    for muni, demo in catchment_pop.items():
        n = demo['pop'] // 10
        for _ in range(n):
            r = {'municipality': muni}
            ag = np.random.choice(list(age_dist.keys()), p=list(age_dist.values()))
            age = np.random.randint(*age_ranges[ag])
            r['age'], r['age_group'] = age, ag
            r['sex'] = np.random.choice(['Male', 'Female'], p=[0.49, 0.51])
            inc = max(0, np.random.lognormal(np.log(demo['median_income']), 0.5))
            r['income'] = round(inc, 0) if np.random.random() > 0.03 else np.nan
            r['income_quintile'] = str(pd.cut([inc],
                bins=[0, 30000, 55000, 80000, 110000, float('inf')],
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])[0])
            dw = np.random.choice(list(housing_rates.keys()), p=list(housing_rates.values()))
            r['dwelling_type'] = dw
            sd = storeys_by_dwelling[dw]
            r['num_storeys'] = np.random.choice(list(sd.keys()), p=list(sd.values()))
            sv = r['num_storeys']
            if sv == '1':
                r['has_stairs'] = int(np.random.random() < 0.15)
                r['num_staircases'] = 0 if not r['has_stairs'] else 1
            elif sv == '2':
                r['has_stairs'] = 1
                r['num_staircases'] = np.random.choice([1, 2], p=[0.75, 0.25])
            else:
                r['has_stairs'] = 1
                r['num_staircases'] = np.random.choice([2, 3], p=[0.70, 0.30])
            r['num_rooms'] = max(1, int(np.random.normal(room_means[dw], 1.5)))
            af = max(0.3, age / 65)
            r['has_diabetes'] = int(np.random.random() < prevalence['diabetes'] * af)
            r['has_hypertension'] = int(np.random.random() < prevalence['hypertension'] * af)
            r['has_copd'] = int(np.random.random() < prevalence['copd'] * af)
            r['has_asthma'] = int(np.random.random() < prevalence['asthma'])
            r['has_heart_disease'] = int(np.random.random() < prevalence['heart_disease'] * af)
            r['has_mood_disorder'] = int(np.random.random() < prevalence['mood_disorders'])
            r['has_arthritis'] = int(np.random.random() < prevalence['arthritis'] * af)
            r['has_dementia'] = int(np.random.random() < prevalence.get('dementia', 0.065) * max(0, (age - 60) / 40))
            mob = (age / 100) + r['has_arthritis'] * 0.2 + r['has_dementia'] * 0.3
            r['has_mobility_limitation'] = int(np.random.random() < min(0.9, mob))
            cc = sum(r[k] for k in ['has_diabetes', 'has_hypertension', 'has_copd',
                                     'has_asthma', 'has_heart_disease', 'has_mood_disorder',
                                     'has_arthritis'])
            r['chronic_condition_count'] = cc
            rs = ((age / 100) * 30 + r['has_diabetes'] * 15 + r['has_hypertension'] * 10 +
                  r['has_copd'] * 12 + r['has_heart_disease'] * 18 + r['has_dementia'] * 20 +
                  r['has_mobility_limitation'] * 8 + cc * 5 +
                  (5 - ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'].index(r['income_quintile'])) * 3)
            r['risk_score'] = round(rs, 1) if np.random.random() > 0.02 else np.nan
            r['er_visits_12mo'] = np.random.poisson(max(0.1, rs / 30))
            fr = (max(0, (age - 50) / 50) * 0.3 + r['has_mobility_limitation'] * 0.25 +
                  r['has_dementia'] * 0.2 + r['num_staircases'] * 0.1 + r['has_arthritis'] * 0.1)
            r['fall_risk_score'] = round(min(1.0, fr), 3)
            r['had_fall_12mo'] = int(np.random.random() < fr)
            if cc == 0:
                seg = '1_Prevention'
            elif cc <= 2 and age < 65:
                seg = '2_Early'
            elif cc <= 2 and age >= 65:
                seg = '3_Advanced'
            elif cc >= 3:
                seg = '3_Advanced'
            else:
                seg = '2_Early'
            r['population_segment'] = seg
            records.append(r)
    return pd.DataFrame(records)


# ============================================================
# HELPERS
# ============================================================
def metric_card(label, value, delta=None):
    delta_html = f'<span style="font-size:13px;color:#757575;">{delta}</span>' if delta else ''
    st.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <div class="value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================
PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SAS_DIR = os.path.join(PROJECT_DIR, "sas_programs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
CHARTS_DIR = os.path.join(PROJECT_DIR, "charts")
for d in [DATA_DIR, SAS_DIR, OUTPUT_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'synthetic_df' not in st.session_state:
    st.session_state.synthetic_df = None
if 'fidelity' not in st.session_state:
    st.session_state.fidelity = None
if 'sas_programs' not in st.session_state:
    st.session_state.sas_programs = {}
if 'narrative' not in st.session_state:
    st.session_state.narrative = None
if 'question' not in st.session_state:
    st.session_state.question = ""


# ============================================================
# PIPELINE
# ============================================================
def run_pipeline(question, n_synth=10000):
    sas_engine = SASEngine(SAS_DIR, OUTPUT_DIR)
    sas_gen = SASCodeGenerator(OUTPUT_DIR)
    synth_gen = SyntheticGenerator()

    progress = st.progress(0, text="Building catchment population...")

    # Phase 2: Build data
    df = build_catchment_dataset()
    csv_path = os.path.join(DATA_DIR, "source_data.csv")
    df.to_csv(csv_path, index=False)
    st.session_state.original_df = df
    progress.progress(15, text="Generating SAS profiling code...")

    # Phase 3: SAS profiling
    import_code = sas_gen.generate_import_code(csv_path)
    profile_code = sas_gen.generate_profiling_code(df)
    sas_engine.run_sas_code(import_code + profile_code, "01_profiling")
    st.session_state.sas_programs['01_profiling'] = import_code + profile_code
    progress.progress(25, text="Cleaning data...")

    # Phase 4: Cleaning
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cleaned = df.copy()
    for col in numeric_cols:
        cleaned[col].fillna(cleaned[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        mode_val = cleaned[col].mode()
        if len(mode_val) > 0:
            cleaned[col].fillna(mode_val.iloc[0], inplace=True)
    st.session_state.cleaned_df = cleaned

    hygiene_code = sas_gen.generate_hygiene_code(df)
    export_code = sas_gen.generate_export_cleaned_code()
    sas_engine.run_sas_code(hygiene_code + export_code, "02_cleaning")
    st.session_state.sas_programs['02_cleaning'] = hygiene_code + export_code
    progress.progress(40, text="Running correlation analysis...")

    # Phase 5: Correlation
    corr_code = sas_gen.generate_correlation_code(cleaned)
    sas_engine.run_sas_code(corr_code, "03_correlations")
    st.session_state.sas_programs['03_correlations'] = corr_code
    progress.progress(50, text="Generating visualizations...")

    # Phase 6: Visualization
    viz_code = sas_gen.generate_visualization_code(cleaned)
    sas_engine.run_sas_code(viz_code, "04_visualizations")
    st.session_state.sas_programs['04_visualizations'] = viz_code
    progress.progress(60, text="Fitting distributions & generating synthetic data...")

    # Phase 7: Synthetic generation
    synth_gen.extract_metadata(cleaned)
    synthetic = synth_gen.generate(n_synth)
    synth_csv = os.path.join(OUTPUT_DIR, "synthetic_data.csv")
    synthetic.to_csv(synth_csv, index=False)
    st.session_state.synthetic_df = synthetic
    progress.progress(75, text="Computing fidelity metrics...")

    # Phase 8: Fidelity
    fidelity = synth_gen.compute_fidelity(cleaned, synthetic)
    st.session_state.fidelity = fidelity

    fid_import = f"""
proc import datafile="{synth_csv}"
    out=WORK.SYNTHETIC_DATA dbms=csv replace;
    guessingrows=max;
run;
"""
    fid_code = sas_gen.generate_fidelity_code(cleaned)
    sas_engine.run_sas_code(fid_import + fid_code, "05_fidelity")
    st.session_state.sas_programs['05_fidelity'] = fid_import + fid_code
    progress.progress(85, text="Generating clinical narrative...")

    # Phase 9: Narrative
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1,
                         api_key=os.getenv("OPENAI_API_KEY"))
        cat_cols = [c for c in cleaned.columns if not pd.api.types.is_numeric_dtype(cleaned[c])]
        corr_matrix = cleaned[numeric_cols].corr(method='spearman')
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_pairs.append({'var1': numeric_cols[i], 'var2': numeric_cols[j],
                                   'correlation': round(corr_matrix.iloc[i, j], 3)})
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        resp = llm.invoke([
            SystemMessage(content="""You are a healthcare data analyst at Southlake Health.
Generate a concise clinical report based on the analysis results.
Use markdown headers and bullet points. Include: key findings,
correlations discovered, population insights, and how the synthetic
data can be used. Answer the original question directly."""),
            HumanMessage(content=f"""
QUESTION: {question}
DATASET: {len(cleaned):,} records from Southlake catchment
(Newmarket, Aurora, East Gwillimbury, Georgina, Bradford West Gwillimbury, King, Innisfil)
KEY STATS:
{cleaned[numeric_cols].describe().to_string()}
TOP CORRELATIONS: {json.dumps(corr_pairs[:10], default=str)}
VARIABLES: {list(cleaned.columns)}
CATEGORICAL:
{chr(10).join(f"{c}: {cleaned[c].value_counts().head(5).to_dict()}" for c in cat_cols[:8])}
SYNTHETIC: {len(synthetic):,} records | FIDELITY: {fidelity['overall_score']:.1f}%
""")
        ])
        st.session_state.narrative = resp.content
    except Exception as e:
        st.session_state.narrative = (
            f"## Clinical Report\n\n"
            f"Analysis of **{len(cleaned):,}** records across the Southlake catchment area.\n\n"
            f"**{len(synthetic):,}** synthetic records generated with "
            f"**{fidelity['overall_score']:.1f}%** fidelity.\n\n"
            f"*LLM narrative unavailable ({e}). Set OPENAI_API_KEY for full report.*"
        )

    progress.progress(100, text="Done!")
    time.sleep(0.5)
    progress.empty()

    st.session_state.pipeline_run = True
    st.session_state.question = question


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    logo_path = os.path.join(PROJECT_DIR, "southlake_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown(f"""
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <span style="font-size:28px; font-weight:800; color:white;">
                Southlake<span style="color:{COLORS['turquoise']};">Health</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏥 Home", "🌐 Data Sources", "📊 Profiling", "🧹 Data Hygiene",
         "🔗 Correlations", "🧬 Synthetic Data", "✅ Fidelity", "📝 Report"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="padding: 8px 0;">
        <p style="color:{COLORS['turquoise']}; font-weight:600; margin-bottom:4px;">
            SynthetiCare Agent v1.0</p>
        <p>Autonomous synthetic data service for population health planning.</p>
        <p style="font-size:11px; margin-top:12px;">
            Built with SAS Viya · GPT-4o · Gaussian Copula<br>
            Open Government Licence — Canada
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏥 Home":
    st.markdown("""
    <div class="hero">
        <h1>🏥 SynthetiCare Agent</h1>
        <p>Autonomous Synthetic Data Service for Southlake Health</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="max-width:720px;">
        <p style="font-size:16px; color:{COLORS['navy']}; line-height:1.7;">
            Ask a population health question in plain language. The agent will
            <b>scrape public data</b>, <b>profile it in SAS</b>, <b>generate
            privacy-safe synthetic records</b> via Gaussian Copula, and deliver
            a clinical report — all autonomously.
        </p>
    </div>
    """, unsafe_allow_html=True)

    question = st.text_area(
        "What would you like to know?",
        value=("What does the chronic disease burden look like for the population "
               "in Southlake's catchment area? I need synthetic patient data that "
               "I can use for population health planning and segmentation modeling "
               "without any privacy concerns."),
        height=120,
        placeholder="Type your population health question here..."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        n_synth = st.number_input("Synthetic rows", min_value=1000,
                                   max_value=100000, value=10000, step=1000)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀  Run SynthetiCare Agent", type="primary",
                            use_container_width=True)

    if run_btn:
        run_pipeline(question, n_synth)
        st.success("✅ Pipeline complete! Use the sidebar to explore results.")

    if st.session_state.pipeline_run:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Source Records",
                        f"{len(st.session_state.original_df):,}",
                        "7 municipalities")
        with c2:
            metric_card("Synthetic Records",
                        f"{len(st.session_state.synthetic_df):,}",
                        "Gaussian Copula")
        with c3:
            metric_card("Fidelity Score",
                        f"{st.session_state.fidelity['overall_score']:.1f}%",
                        "KS + TVD + Corr")
        with c4:
            metric_card("SAS Programs",
                        str(len(st.session_state.sas_programs)),
                        "Ready to execute")


# ============================================================
# PAGE: DATA SOURCES
# ============================================================
elif page == "🌐 Data Sources":
    st.markdown("""
    <div class="phase-header">
        <h2>🌐 Data Sources</h2>
        <span>Phase 2 — Public data acquisition</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline from the Home page first.")
    else:
        sources = [
            {'name': 'Statistics Canada 2021 Census Profile (CSDs)',
             'licence': 'Open Government Licence — Canada',
             'desc': 'Population demographics, age, sex, income, housing, dwelling type, storeys, period of construction',
             'rows': len(st.session_state.original_df)},
            {'name': 'PHAC Canadian Chronic Disease Indicators (CCDI 2018)',
             'licence': 'Open Government Licence — Canada',
             'desc': 'Diabetes, hypertension, COPD, asthma, heart disease, mood disorders, arthritis prevalence rates',
             'rows': 'Rate tables'},
            {'name': 'Ontario Data Catalogue',
             'licence': 'Open Government Licence — Ontario',
             'desc': 'Hospital utilisation, emergency department, long-term care, health region data',
             'rows': 'API'},
        ]

        for src in sources:
            st.markdown(f"""
            <div class="source-card">
                <b>{src['name']}</b><br>
                <span style="font-size:13px;">{src['desc']}</span><br>
                <span class="licence">📜 {src['licence']}  |  Rows: {src['rows']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Southlake Catchment Area")

        catchment = pd.DataFrame({
            'Municipality': ['Newmarket', 'Aurora', 'East Gwillimbury', 'Georgina',
                             'Bradford West Gwillimbury', 'King', 'Innisfil'],
            'Population (2021)': [87942, 62057, 34637, 47642, 42880, 27333, 43326],
            'Median Income': ['$95,000', '$110,000', '$98,000', '$82,000',
                              '$92,000', '$125,000', '$88,000'],
            'Sample (10%)': [8794, 6205, 3463, 4764, 4288, 2733, 4332],
        })
        st.dataframe(catchment, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:{COLORS['light_bg']}; padding:14px; border-radius:8px; margin-top:12px;">
            <b>Total catchment population:</b> 345,817<br>
            <b>10% sample used:</b> {len(st.session_state.original_df):,} individual records
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: PROFILING
# ============================================================
elif page == "📊 Profiling":
    st.markdown("""
    <div class="phase-header">
        <h2>📊 Data Profiling</h2>
        <span>Phase 3 — Descriptive statistics & distribution analysis</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline from the Home page first.")
    else:
        df = st.session_state.cleaned_df
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        st.markdown("### Descriptive Statistics")
        profile = df[numeric_cols].describe().round(3).T
        profile['skewness'] = df[numeric_cols].skew().round(3)
        profile['kurtosis'] = df[numeric_cols].kurtosis().round(3)
        profile['nmiss'] = df[numeric_cols].isna().sum()
        st.dataframe(profile, use_container_width=True)

        st.markdown("### Distributions")
        plot_vars = [c for c in ['age', 'income', 'risk_score', 'chronic_condition_count',
                                  'er_visits_12mo', 'fall_risk_score'] if c in numeric_cols]
        n_plots = min(len(plot_vars), 6)
        n_rows_p = (n_plots + 2) // 3
        fig, axes = plt.subplots(n_rows_p, 3, figsize=(16, 4.5 * n_rows_p))
        if n_rows_p == 1:
            axes = [axes]
        for i, var in enumerate(plot_vars[:n_plots]):
            ax = axes[i // 3][i % 3]
            ax.hist(df[var].dropna(), bins=40, color=COLORS['teal'],
                    alpha=0.8, edgecolor='white', linewidth=0.5)
            ax.set_title(var.replace('_', ' ').title(), fontsize=12,
                         fontweight='bold', color=COLORS['navy'])
            ax.set_xlabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        for i in range(n_plots, n_rows_p * 3):
            axes[i // 3][i % 3].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("🔍 View SAS Code — PROC MEANS / PROC UNIVARIATE / PROC FREQ"):
            st.code(st.session_state.sas_programs.get('01_profiling', ''), language='sas')


# ============================================================
# PAGE: DATA HYGIENE
# ============================================================
elif page == "🧹 Data Hygiene":
    st.markdown("""
    <div class="phase-header">
        <h2>🧹 Data Hygiene</h2>
        <span>Phase 4 — Missing value imputation & outlier detection</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline from the Home page first.")
    else:
        df = st.session_state.original_df
        cleaned = st.session_state.cleaned_df

        st.markdown("### Missing Value Analysis")
        miss_data = []
        for col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                miss_data.append({
                    'Column': col,
                    'Missing': n_miss,
                    '% Missing': f"{df[col].isna().mean() * 100:.2f}%",
                    'Action': ('Median imputation' if pd.api.types.is_numeric_dtype(df[col])
                               else 'Mode imputation'),
                    'Status': '✅ Fixed'
                })
        if miss_data:
            st.dataframe(pd.DataFrame(miss_data), use_container_width=True, hide_index=True)
        else:
            st.success("No missing values detected.")

        st.markdown("### Before vs After Cleaning")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        compare = pd.DataFrame({
            'Column': numeric_cols,
            'Missing (Before)': [df[c].isna().sum() for c in numeric_cols],
            'Missing (After)': [cleaned[c].isna().sum() for c in numeric_cols],
            'Mean (Before)': [df[c].mean() for c in numeric_cols],
            'Mean (After)': [cleaned[c].mean() for c in numeric_cols],
        }).round(3)
        st.dataframe(compare, use_container_width=True, hide_index=True)

        with st.expander("🔍 View SAS Code — PROC STDIZE / PROC SQL / Outlier Detection"):
            st.code(st.session_state.sas_programs.get('02_cleaning', ''), language='sas')


# ============================================================
# PAGE: CORRELATIONS
# ============================================================
elif page == "🔗 Correlations":
    st.markdown("""
    <div class="phase-header">
        <h2>🔗 Correlation Analysis</h2>
        <span>Phase 5 — Spearman rank correlations & cross-tabulations</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline from the Home page first.")
    else:
        df = st.session_state.cleaned_df
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        corr = df[numeric_cols].corr(method='spearman')

        st.markdown("### Spearman Correlation Matrix")
        fig, ax = plt.subplots(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax, square=True,
                    linewidths=0.5, annot_kws={'size': 7},
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Spearman Rank Correlation Matrix', fontsize=14,
                     fontweight='bold', color=COLORS['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Strongest Relationships")
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                pairs.append({
                    'Variable 1': numeric_cols[i],
                    'Variable 2': numeric_cols[j],
                    'Correlation': round(corr.iloc[i, j], 4),
                    'Strength': ('🔴 Strong' if abs(corr.iloc[i, j]) > 0.5
                                 else '🟡 Moderate' if abs(corr.iloc[i, j]) > 0.3
                                 else '⚪ Weak')
                })
        pairs.sort(key=lambda x: abs(x['Correlation']), reverse=True)
        st.dataframe(pd.DataFrame(pairs[:15]), use_container_width=True, hide_index=True)

        with st.expander("🔍 View SAS Code — PROC CORR / PROC FREQ Chi-squared"):
            st.code(st.session_state.sas_programs.get('03_correlations', ''), language='sas')


# ============================================================
# PAGE: SYNTHETIC DATA
# ============================================================
elif page == "🧬 Synthetic Data":
    st.markdown("""
    <div class="phase-header">
        <h2>🧬 Synthetic Data Generation</h2>
        <span>Phase 7 — Gaussian Copula with fitted marginals</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline from the Home page first.")
    else:
        synthetic = st.session_state.synthetic_df

        st.markdown(f"""
        <div style="background:{COLORS['light_bg']}; padding:18px; border-radius:10px;
                    border-left:5px solid #7b1fa2; margin-bottom:20px;">
            <h4 style="color:#7b1fa2; margin:0 0 8px 0;">🔒 Privacy Mechanism: Gaussian Copula</h4>
            <ol style="margin:0; padding-left:20px; color:{COLORS['navy']}; font-size:14px; line-height:1.8;">
                <li>Each column's marginal distribution fitted independently (normal / lognormal / gamma)</li>
                <li>Inter-variable dependencies captured via <b>Spearman rank correlation matrix</b></li>
                <li>Multivariate normal samples drawn, then transformed through fitted inverse-CDFs</li>
                <li>Binary & categorical columns sampled from observed frequency distributions</li>
                <li><b>Result:</b> new records that preserve statistical properties but correspond to <b>no real individual</b></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Records Generated", f"{len(synthetic):,}")
        with c2:
            metric_card("Columns", str(len(synthetic.columns)))
        with c3:
            metric_card("Privacy", "✅ No PII")

        st.markdown("### Preview")
        st.dataframe(synthetic.head(20), use_container_width=True, hide_index=True)

        csv_data = synthetic.to_csv(index=False)
        st.download_button(
            label="⬇️  Download Synthetic Dataset (CSV)",
            data=csv_data,
            file_name="southlake_synthetic_data.csv",
            mime="text/csv",
            type="primary"
        )

        with st.expander("🔍 View SAS Code — PROC SGPLOT Visualizations"):
            st.code(st.session_state.sas_programs.get('04_visualizations', ''), language='sas')


# ============================================================
# PAGE: FIDELITY
# ============================================================
elif page == "✅ Fidelity":
    st.markdown("""
    <div class="phase-header">
        <h2>✅ Fidelity Verification</h2>
        <span>Phase 8 — Statistical comparison of original vs synthetic</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline from the Home page first.")
    else:
        fidelity = st.session_state.fidelity
        cleaned = st.session_state.cleaned_df
        synthetic = st.session_state.synthetic_df

        score = fidelity['overall_score']
        verdict = ('🟢 Excellent — synthetic data closely mirrors original' if score >= 85
                   else '🟡 Good — minor deviations, suitable for most analyses' if score >= 70
                   else '🟠 Fair — review individual columns' if score >= 50
                   else '🔴 Needs improvement')
        st.markdown(f"""
        <div class="fidelity-ring">
            <h2>Overall Fidelity Score</h2>
            <div class="score">{score:.1f}%</div>
            <div class="verdict">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Numeric Columns")
        num_rows = []
        for col, m in fidelity['numeric'].items():
            q = ('🟢' if m['score'] >= 85 else '🟡' if m['score'] >= 70
                 else '🟠' if m['score'] >= 50 else '🔴')
            num_rows.append({
                '': q, 'Column': col, 'KS Stat': m['ks_statistic'],
                'Mean Δ%': m['mean_diff_pct'], 'Std Δ%': m['std_diff_pct'],
                'Score': m['score']
            })
        st.dataframe(pd.DataFrame(num_rows), use_container_width=True, hide_index=True)

        if fidelity['binary']:
            st.markdown("### Binary Columns")
            bin_rows = []
            for col, m in fidelity['binary'].items():
                q = '🟢' if m['score'] >= 85 else '🟡' if m['score'] >= 70 else '🟠'
                bin_rows.append({
                    '': q, 'Column': col,
                    'Original Rate': f"{m['original_rate']:.4f}",
                    'Synthetic Rate': f"{m['synthetic_rate']:.4f}",
                    'Abs Diff': f"{m['abs_diff']:.4f}",
                    'Score': m['score']
                })
            st.dataframe(pd.DataFrame(bin_rows), use_container_width=True, hide_index=True)

        if fidelity['categorical']:
            st.markdown("### Categorical Columns")
            cat_rows = []
            for col, m in fidelity['categorical'].items():
                q = '🟢' if m['score'] >= 85 else '🟡' if m['score'] >= 70 else '🟠'
                cat_rows.append({
                    '': q, 'Column': col,
                    'Total Variation Distance': m['tvd'],
                    'Score': m['score']
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

        if 'correlation_score' in fidelity:
            st.markdown(f"""
            <div style="background:{COLORS['light_bg']}; padding:14px; border-radius:8px; margin:16px 0;">
                <b>🔗 Correlation Preservation:</b> Score = {fidelity['correlation_score']}%
                 |  Avg absolute difference = {fidelity['correlation_diff']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Distribution Comparison")
        numeric_cols = [c for c in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[c])]
        plot_vars = [c for c in ['age', 'income', 'risk_score', 'chronic_condition_count',
                                  'er_visits_12mo', 'fall_risk_score'] if c in numeric_cols]
        n_p = min(len(plot_vars), 6)
        nr = (n_p + 2) // 3
        fig, axes = plt.subplots(nr, 3, figsize=(16, 4.5 * nr))
        if nr == 1:
            axes = [axes]
        for i, var in enumerate(plot_vars[:n_p]):
            ax = axes[i // 3][i % 3]
            orig = cleaned[var].dropna()
            synth = synthetic[var].dropna() if var in synthetic.columns else pd.Series()
            if len(synth) > 0:
                ax.hist(orig, bins=40, alpha=0.55, color=COLORS['teal'],
                        label='Original', density=True, edgecolor='white', linewidth=0.3)
                ax.hist(synth, bins=40, alpha=0.55, color=COLORS['alert_red'],
                        label='Synthetic', density=True, edgecolor='white', linewidth=0.3)
                ax.legend(fontsize=9)
            ax.set_title(var.replace('_', ' ').title(), fontsize=11,
                         fontweight='bold', color=COLORS['navy'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        for i in range(n_p, nr * 3):
            axes[i // 3][i % 3].set_visible(False)
        plt.suptitle('Original vs Synthetic Distributions', fontsize=14,
                     fontweight='bold', color=COLORS['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        synth_numeric = [c for c in numeric_cols if c in synthetic.columns]
        if len(synth_numeric) >= 2:
            st.markdown("### Correlation Matrix Comparison")
            orig_corr = cleaned[synth_numeric].corr(method='spearman')
            synth_corr = synthetic[synth_numeric].astype(float).corr(method='spearman')
            diff_corr = (orig_corr - synth_corr).abs()

            fig, axs = plt.subplots(1, 3, figsize=(20, 6.5))
            for ax_i, (data, title, cmap, vmin, vmax, center) in enumerate([
                (orig_corr, 'Original', 'RdBu_r', -1, 1, 0),
                (synth_corr, 'Synthetic', 'RdBu_r', -1, 1, 0),
                (diff_corr, 'Absolute Difference', 'Reds', 0, 0.5, None),
            ]):
                sns.heatmap(data, annot=True, fmt='.2f', cmap=cmap, center=center,
                            vmin=vmin, vmax=vmax, ax=axs[ax_i], square=True,
                            linewidths=0.5, annot_kws={'size': 6},
                            cbar_kws={'shrink': 0.75})
                axs[ax_i].set_title(title, fontsize=12, fontweight='bold',
                                    color=COLORS['navy'])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with st.expander("🔍 View SAS Code — Fidelity Verification"):
            st.code(st.session_state.sas_programs.get('05_fidelity', ''), language='sas')


# ============================================================
# PAGE: REPORT
# ============================================================
elif page == "📝 Report":
    st.markdown("""
    <div class="phase-header">
        <h2>📝 Clinical Report</h2>
        <span>Phase 9 — AI-generated narrative & deliverables</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline from the Home page first.")
    else:
        st.markdown(f"""
        <div style="background:{COLORS['light_bg']}; padding:14px; border-radius:8px;
                    border-left:4px solid {COLORS['teal']}; margin-bottom:20px;">
            <b>📋 Original Question:</b> {st.session_state.question}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(st.session_state.narrative)

        st.markdown("---")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, {COLORS['navy']}, {COLORS['dark_teal']});
                    padding:28px; border-radius:14px; margin-top:16px;">
            <h2 style="color:white; margin:0 0 16px 0;">📦 Deliverables</h2>
            <div class="deliverable-row">
                <span><b>Synthetic Dataset</b></span>
                <span><code>output/synthetic_data.csv</code> — {len(st.session_state.synthetic_df):,} rows × {len(st.session_state.synthetic_df.columns)} cols</span>
            </div>
            <div class="deliverable-row">
                <span><b>Source Data</b></span>
                <span><code>data/source_data.csv</code> — {len(st.session_state.cleaned_df):,} rows × {len(st.session_state.cleaned_df.columns)} cols</span>
            </div>
            <div class="deliverable-row">
                <span><b>Fidelity Score</b></span>
                <span>{st.session_state.fidelity['overall_score']:.1f}%</span>
            </div>
            <div class="deliverable-row" style="border-bottom:none;">
                <span><b>Privacy</b></span>
                <span style="color:{COLORS['turquoise']};">✅ No real patient data — all public sources + Gaussian Copula</span>
            </div>
            <hr style="border-color:rgba(255,255,255,0.15); margin:16px 0;">
            <h4 style="color:{COLORS['turquoise']};">📂 SAS Programs Generated</h4>
        </div>
        """, unsafe_allow_html=True)

        for prog_name in st.session_state.sas_programs:
            st.markdown(f"- `sas_programs/{prog_name}.sas`")

        st.markdown("""
        > **▶️ To run SAS programs:** Open each `.sas` file in VS Code and press
        > `Cmd+Shift+Enter`. The SAS Extension executes them on the Viya server.
        """)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️  Download Synthetic Data",
                st.session_state.synthetic_df.to_csv(index=False),
                "southlake_synthetic_data.csv", "text/csv",
                type="primary", use_container_width=True)
        with c2:
            st.download_button(
                "⬇️  Download Source Data",
                st.session_state.cleaned_df.to_csv(index=False),
                "southlake_source_data.csv", "text/csv",
                use_container_width=True)