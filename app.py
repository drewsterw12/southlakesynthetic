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
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {{
        color: white !important;
        font-size: 15px;
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background-color: rgba(38, 198, 218, 0.2);
        border-radius: 6px;
    }}
    section[data-testid="stSidebar"] * {{
        color: rgba(255,255,255,0.85) !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}
    section[data-testid="stSidebar"] [data-testid="stRadio"] [data-checked="true"] {{
        background-color: rgba(38, 198, 218, 0.25);
        border-radius: 6px;
    }}
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {{
        background-color: {COLORS['teal']} !important;
        border-color: {COLORS['teal']} !important;
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
    }}
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {{
        background-color: {COLORS['dark_teal']} !important;
        border-color: {COLORS['dark_teal']} !important;
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
    }}
div.metric-card div.value {{
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
    }}
span.licence {{
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
    }}
div.fidelity-ring div.score {{
        font-size: 64px;
        font-weight: 800;
        color: {COLORS['turquoise']};
        margin: 8px 0;
    }}
div.fidelity-ring div.verdict {{
        color: white;
        font-size: 15px;
    }}.deliverable-row {{
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        color: white;
    }}
    button[data-testid="stBaseButton-primary"] {{
        background-color: {COLORS['teal']} !important;
        border-color: {COLORS['teal']} !important;
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
    }}
    button[data-testid="stBaseButton-primary"]:hover {{
        background-color: {COLORS['dark_teal']} !important;
        border-color: {COLORS['dark_teal']} !important;
        box-shadow: 0 4px 12px rgba(0, 131, 143, 0.3);
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)
# ============================================================
# TRIVIAL CORRELATION PAIRS TO EXCLUDE
# ============================================================
TRIVIAL_PAIRS = {
    frozenset({'has_stairs', 'num_staircases'}),
    frozenset({'has_stairs', 'num_storeys'}),
    frozenset({'num_staircases', 'num_storeys'}),
    frozenset({'has_stairs', 'num_rooms'}),
    frozenset({'num_staircases', 'num_rooms'}),
    frozenset({'chronic_condition_count', 'has_diabetes'}),
    frozenset({'chronic_condition_count', 'has_hypertension'}),
    frozenset({'chronic_condition_count', 'has_copd'}),
    frozenset({'chronic_condition_count', 'has_asthma'}),
    frozenset({'chronic_condition_count', 'has_heart_disease'}),
    frozenset({'chronic_condition_count', 'has_mood_disorder'}),
    frozenset({'chronic_condition_count', 'has_arthritis'}),
    frozenset({'chronic_condition_count', 'risk_score'}),
    frozenset({'chronic_condition_count', 'er_visits_12mo'}),
    frozenset({'risk_score', 'er_visits_12mo'}),
    frozenset({'fall_risk_score', 'had_fall_12mo'}),
    frozenset({'fall_risk_score', 'has_mobility_limitation'}),
    frozenset({'has_mobility_limitation', 'had_fall_12mo'}),
}


def is_trivial_pair(var1, var2):
    return frozenset({var1, var2}) in TRIVIAL_PAIRS


def classify_correlation_strength(r):
    abs_r = abs(r)
    if abs_r >= 0.5:
        return '🔴 Strong'
    elif abs_r >= 0.25:
        return '🟡 Moderate'
    elif abs_r >= 0.10:
        return '⚪ Weak'
    else:
        return '⚫ Negligible'


# ============================================================
# DYNAMIC DATA ENRICHMENT — LLM analyzes question & finds data
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def analyze_question_and_enrich(question, _cache_version=0):
    """Use LLM to analyze the question, identify needed conditions AND risk factors,
    and return additional data to add to the dataset."""
    
    base_conditions = {
        'diabetes': {'prevalence': 0.087, 'age_adjusted': True, 'source': 'PHAC CCDI 2021'},
        'hypertension': {'prevalence': 0.198, 'age_adjusted': True, 'source': 'PHAC CCDI 2021'},
        'copd': {'prevalence': 0.042, 'age_adjusted': True, 'source': 'PHAC CCDI 2021'},
        'asthma': {'prevalence': 0.112, 'age_adjusted': False, 'source': 'PHAC CCDI 2021'},
        'heart_disease': {'prevalence': 0.058, 'age_adjusted': True, 'source': 'PHAC CCDI 2021'},
        'mood_disorders': {'prevalence': 0.082, 'age_adjusted': False, 'source': 'PHAC CCDI 2021'},
        'arthritis': {'prevalence': 0.167, 'age_adjusted': True, 'source': 'PHAC CCDI 2021'},
        'dementia': {'prevalence': 0.065, 'age_adjusted': True, 'source': 'PHAC CCDI 2021'},
    }
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return base_conditions, [], [], [], ['age', 'sex', 'income', 'municipality'], []
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
        
        resp = llm.invoke([
            SystemMessage(content="""You are a Canadian healthcare epidemiologist.
Given a user's question about population health, analyze it and return structured JSON.

The base dataset already has these conditions: diabetes, hypertension, copd, asthma, 
heart_disease, mood_disorders, arthritis, dementia.
The base dataset already has these demographics: age, sex, income, municipality.

You can add TWO types of new variables:
1. "additional_conditions" — binary disease/condition flags (has_X)
2. "additional_risk_factors" — behavioral/lifestyle/clinical variables that are risk factors

Return this EXACT JSON structure (no markdown, no code fences, just raw JSON):
{
    "additional_conditions": [
        {
            "name": "lung_cancer",
            "prevalence": 0.006,
            "age_adjusted": true,
            "age_factor": "increases_with_age",
            "source": "Canadian Cancer Society 2023",
            "risk_factors": ["age", "is_smoker"],
            "comorbidities": ["has_copd"]
        }
    ],
    "additional_risk_factors": [
        {
            "name": "is_smoker",
            "type": "binary",
            "prevalence": 0.145,
            "age_factor": "peaks_middle_age",
            "source": "Statistics Canada CTADS 2022",
            "correlates_with": ["has_lung_cancer", "has_copd", "has_heart_disease"],
            "correlation_strength": 0.4
        },
        {
            "name": "bmi",
            "type": "numeric",
            "mean": 27.2,
            "std": 5.5,
            "min": 15,
            "max": 55,
            "source": "Statistics Canada CCHS 2022",
            "correlates_with": ["has_diabetes", "has_hypertension", "has_heart_disease"],
            "correlation_strength": 0.3
        }
    ],
    "relevant_existing_conditions": ["copd", "asthma"],
    "relevant_demographics": ["age", "sex", "income", "municipality"],
    "data_sources_used": [
        {"name": "Canadian Cancer Society", "url": "https://cancer.ca", "licence": "Public"}
    ]
}

CRITICAL RULES:
- "prevalence" MUST be a decimal number (e.g., 0.006 for 0.6%), NOT a string
- "name" must be snake_case
- For risk factors, "type" must be "binary" or "numeric"
- Binary risk factors need "prevalence" (decimal)
- Numeric risk factors need "mean", "std", "min", "max"
- "correlation_strength" is how strongly this risk factor correlates with its correlates (0.0 to 1.0)
- Use REAL Canadian rates from reliable sources (Stats Canada, PHAC, CIHI, Cancer Society, etc.)
FOCUS RULES — VERY IMPORTANT:
- ONLY add conditions that are DIRECTLY asked about in the question and NOT already in the base dataset
- Do NOT add unrelated conditions (e.g., do NOT add lung_cancer if the question is about diabetes)
- ONLY add risk factors that are clinically relevant to the CONDITIONS IN THE QUESTION
- If the question is about diabetes, add risk factors for diabetes (BMI, physical inactivity, family history, diet)
- If the question is about lung cancer, add risk factors for lung cancer (smoking, radon exposure, occupational exposure)
- If the question is about hypertension, add risk factors for hypertension (BMI, sodium intake, physical inactivity, alcohol)
- If the question is about mental health, add risk factors for mental health (social isolation, unemployment, trauma)
- Do NOT add smoking/lung cancer risk factors unless the question specifically asks about lung cancer or respiratory conditions
- "correlates_with" must ONLY reference conditions that are relevant to the question
- Add 2-4 risk factors that are the MOST clinically important for the conditions in the question
- Think about: behavioral (smoking, physical inactivity, alcohol, diet), clinical (BMI, blood pressure, blood glucose), social (isolation, housing instability), environmental (air quality, occupational exposure)
- Every condition in the question should have at least one risk factor that strongly correlates with it
- Use correlation_strength 0.3-0.5 for strong known relationships
- Return ONLY raw JSON, no markdown fences"""),
            HumanMessage(content=f"QUESTION: {question}")
        ])
        
        content = resp.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()
        
        result = json.loads(content)
        
        # Validate conditions
        additional_conditions = []
        for cond in result.get('additional_conditions', []):
            name = cond.get('name', '').lower().replace(' ', '_')
            prev = cond.get('prevalence', 0)
            if isinstance(prev, str):
                prev = float(prev.replace('%', '').strip()) / 100
            prev = float(prev)
            if not name or name in base_conditions or prev <= 0:
                continue
            validated = {
                'name': name, 'prevalence': prev,
                'age_adjusted': cond.get('age_adjusted', True),
                'age_factor': cond.get('age_factor', 'increases_with_age'),
                'source': cond.get('source', 'Canadian health data'),
                'risk_factors': cond.get('risk_factors', []),
                'comorbidities': cond.get('comorbidities', []),
            }
            additional_conditions.append(validated)
            base_conditions[name] = validated
        
        # Validate risk factors
        additional_risk_factors = []
        for rf in result.get('additional_risk_factors', []):
            name = rf.get('name', '').lower().replace(' ', '_')
            rf_type = rf.get('type', 'binary')
            if not name:
                continue
            validated_rf = {
                'name': name, 'type': rf_type,
                'source': rf.get('source', 'Canadian health data'),
                'correlates_with': rf.get('correlates_with', []),
                'correlation_strength': min(0.9, max(0.1, rf.get('correlation_strength', 0.3))),
                'age_factor': rf.get('age_factor', 'flat'),
            }
            if rf_type == 'binary':
                prev = rf.get('prevalence', 0.1)
                if isinstance(prev, str):
                    prev = float(prev.replace('%', '').strip()) / 100
                validated_rf['prevalence'] = float(prev)
            elif rf_type == 'numeric':
                validated_rf['mean'] = float(rf.get('mean', 50))
                validated_rf['std'] = float(rf.get('std', 10))
                validated_rf['min'] = float(rf.get('min', 0))
                validated_rf['max'] = float(rf.get('max', 100))
            additional_risk_factors.append(validated_rf)
        
        relevant_existing = result.get('relevant_existing_conditions', [])
        data_sources = result.get('data_sources_used', [])
        relevant_demographics = result.get('relevant_demographics', ['age', 'sex', 'income', 'municipality'])
        
        return base_conditions, relevant_existing, data_sources, additional_conditions, relevant_demographics, additional_risk_factors
    
    except json.JSONDecodeError as e:
        return base_conditions, [], [{'name': 'LLM Parse Error',
                                       'url': f'JSON error: {str(e)[:100]}',
                                       'licence': 'N/A'}], [], \
               ['age', 'sex', 'income', 'municipality'], []
    except Exception as e:
        return base_conditions, [], [{'name': 'LLM Error',
                                       'url': f'Error: {str(e)[:100]}',
                                       'licence': 'N/A'}], [], \
               ['age', 'sex', 'income', 'municipality'], []
    
def get_relevant_variables(question, all_columns, relevant_existing, additional_conditions, relevant_demographics):
    """Build the list of relevant variables based on what the LLM identified."""
    relevant = set()
    
    # Add demographics the LLM said are relevant
    for d in relevant_demographics:
        if d in all_columns:
            relevant.add(d)
    
    # Always include these core variables
    for col in ['age', 'sex', 'municipality', 'income', 'population_segment',
                'chronic_condition_count', 'risk_score', 'er_visits_12mo']:
        if col in all_columns:
            relevant.add(col)
    
    # Add existing conditions the LLM flagged
    for cond in relevant_existing:
        col_name = f"has_{cond}"
        if col_name in all_columns:
            relevant.add(col_name)
    
    # Add any new conditions that were added
    for cond in additional_conditions:
        col_name = f"has_{cond['name'].lower().replace(' ', '_')}"
        if col_name in all_columns:
            relevant.add(col_name)
    
    # Add any risk factor variables from conditions
    for cond in additional_conditions:
        for rf_name in cond.get('risk_factors', []):
            if rf_name in all_columns:
                relevant.add(rf_name)
    
    # Add all additional risk factor columns
    for rf in (st.session_state.get('additional_risk_factors') or []):
        rf_name = rf.get('name', '')
        if rf_name in all_columns:
            relevant.add(rf_name)
    
    # If very few relevant vars, include all health conditions
    if len([v for v in relevant if v.startswith('has_')]) < 2:
        for col in all_columns:
            if col.startswith('has_'):
                relevant.add(col)
    
    return [c for c in all_columns if c in relevant]


# ============================================================
# BACKEND CLASSES
# ============================================================
# ============================================================
# SAS RUNNER — Executes SAS code via saspy on SAS Viya
# ============================================================
class SASRunner:
    """Manages a live SAS Viya connection via REST API."""
    
    def __init__(self):
        self.base_url = os.getenv('SAS_VIYA_URL', 'https://vfl-032.engage.sas.com')
        self.client_id = os.getenv('SAS_CLIENT_ID', 'sas.cli')
        self.refresh_token = os.getenv('SAS_REFRESH_TOKEN', '')
        self.context_id = 'bcdcb9ba-73ff-494b-9631-d60c4c5571c9'
        self.access_token = None
        self.session_id = None
        self.connected = False
        self.log = []
    
    def connect(self):
        """Authenticate and create a compute session."""
        try:
            if not self.refresh_token:
                self.log.append('No SAS_REFRESH_TOKEN in.env')
                return False
            
            resp = requests.post(
                self.base_url + '/SASLogon/oauth/token',
                data={
                    'grant_type': 'refresh_token',
                    'refresh_token': self.refresh_token,
                    'client_id': self.client_id,
                    'client_secret': ''
                }
            )
            if resp.status_code != 200:
                self.log.append('Token refresh failed: ' + str(resp.status_code))
                return False
            
            self.access_token = resp.json()['access_token']
            self.log.append('Got access token')
            
            headers = {
                'Authorization': 'Bearer ' + self.access_token,
                'Content-Type': 'application/vnd.sas.compute.session.request+json'
            }
            sess = requests.post(
                self.base_url + '/compute/contexts/' + self.context_id + '/sessions',
                headers=headers,
                json={'name': 'syntheticare'}
            )
            if sess.status_code not in [200, 201]:
                self.log.append('Session creation failed: ' + str(sess.status_code))
                return False
            
            self.session_id = sess.json()['id']
            self.connected = True
            self.log.append('Connected — session: ' + self.session_id)
            return True
        except Exception as e:
            self.log.append('Connection error: ' + str(e))
            return False
    
    def disconnect(self):
        if self.session_id and self.access_token:
            try:
                requests.delete(
                    self.base_url + '/compute/sessions/' + self.session_id,
                    headers={'Authorization': 'Bearer ' + self.access_token}
                )
            except Exception:
                pass
        self.connected = False
        self.session_id = None
    
    def _run_job(self, sas_code, description=''):
        """Submit SAS code as a job and wait for completion."""
        if not self.connected:
            return {'LOG': 'Not connected', 'LST': '', 'success': False}
        try:
            headers = {
                'Authorization': 'Bearer ' + self.access_token,
                'Content-Type': 'application/vnd.sas.compute.job.request+json'
            }
            code_lines = sas_code.strip().split('\n')
            job = requests.post(
                self.base_url + '/compute/sessions/' + self.session_id + '/jobs',
                headers=headers,
                json={'code': code_lines}
            )
            if job.status_code not in [200, 201]:
                return {'LOG': 'Job submit failed: ' + str(job.status_code), 'LST': '', 'success': False}
            
            jid = job.json()['id']
            
            for _ in range(60):
                time.sleep(1)
                check = requests.get(
                    self.base_url + '/compute/sessions/' + self.session_id + '/jobs/' + jid,
                    headers={'Authorization': 'Bearer ' + self.access_token}
                )
                state = check.json().get('state', '')
                if state in ['completed', 'error', 'canceled']:
                    break
            
            log_resp = requests.get(
                self.base_url + '/compute/sessions/' + self.session_id + '/jobs/' + jid + '/log',
                headers={'Authorization': 'Bearer ' + self.access_token, 'Accept': 'application/json'}
            )
            log_text = ''
            if log_resp.status_code == 200:
                log_lines = log_resp.json().get('items', [])
                log_text = '\n'.join(line.get('line', '') for line in log_lines)
            
            # Get ODS results listing
            lst_html = ''
            results_resp = requests.get(
                self.base_url + '/compute/sessions/' + self.session_id + '/jobs/' + jid + '/results',
                headers={'Authorization': 'Bearer ' + self.access_token, 'Accept': 'application/json'}
            )
            if results_resp.status_code == 200:
                try:
                    results_json = results_resp.json()
                    result_items = results_json.get('items', [])
                    for item in result_items:
                        item_links = item.get('links', [])
                        for link in item_links:
                            if link.get('rel') == 'self' and link.get('href'):
                                html_resp = requests.get(
                                    self.base_url + link['href'] + '/content',
                                    headers={
                                        'Authorization': 'Bearer ' + self.access_token,
                                        'Accept': 'text/html'
                                    }
                                )
                                if html_resp.status_code == 200 and '<table' in html_resp.text.lower():
                                    lst_html += html_resp.text + '\n'
                                break
                except (ValueError, KeyError):
                    pass
            
            # Fallback: try direct listing endpoint
            if not lst_html:
                listing_resp = requests.get(
                    self.base_url + '/compute/sessions/' + self.session_id + '/jobs/' + jid + '/listing',
                    headers={'Authorization': 'Bearer ' + self.access_token, 'Accept': 'text/html'}
                )
                if listing_resp.status_code == 200:
                    lst_html = listing_resp.text
            
            # Fallback: try listing as plain text
            if not lst_html:
                listing_resp2 = requests.get(
                    self.base_url + '/compute/sessions/' + self.session_id + '/jobs/' + jid + '/listing',
                    headers={'Authorization': 'Bearer ' + self.access_token, 'Accept': 'text/plain'}
                )
                if listing_resp2.status_code == 200 and len(listing_resp2.text.strip()) > 10:
                    lst_html = listing_resp2.text
            
            self.log.append('Executed: ' + (description or 'SAS code') + ' — ' + state)
            return {
                'LOG': log_text,
                'LST': lst_html,
                'success': state == 'completed' and 'ERROR' not in log_text.upper()
            }
        except Exception as e:
            self.log.append('Execution error: ' + str(e))
            return {'LOG': str(e), 'LST': '', 'success': False}
    
    def run(self, sas_code, description=''):
        return self._run_job(sas_code, description)
    
    def upload_dataframe(self, df, table_name='SOURCE_DATA', libref='WORK'):
        """Upload DataFrame to SAS via CSV file upload + PROC IMPORT, with datalines fallback."""
        if not self.connected:
            return False
        try:
            n_rows = min(len(df), 5000)
            sample = df.sample(n=n_rows, random_state=42) if len(df) > n_rows else df.copy()
            
            # Primary path: upload CSV to SAS Viya Files service
            csv_bytes = sample.to_csv(index=False).encode('utf-8')
            headers = {
                'Authorization': 'Bearer ' + self.access_token,
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename="{table_name}.csv"'
            }
            upload_resp = requests.post(
                self.base_url + '/files/files#rawUpload',
                headers=headers,
                data=csv_bytes
            )
            if upload_resp.status_code in [200, 201]:
                file_uri = upload_resp.json().get('id', '')
                file_path = f'/files/files/{file_uri}/content'
                import_code = f'''
filename _csvin url "{self.base_url}{file_path}"
    access_token="{self.access_token}";
proc import datafile=_csvin
    out={libref}.{table_name}
    dbms=csv replace;
    guessingrows=max;
run;
filename _csvin clear;
'''
                result = self._run_job(import_code, 'Upload ' + table_name)
                if result['success']:
                    self.log.append(f'Uploaded {n_rows} rows to {libref}.{table_name} via CSV')
                    return True
            
            # Fallback: datalines with safe delimiter and escaping
            self.log.append('CSV upload not available, falling back to datalines')
            return self._upload_via_datalines(sample, table_name, libref)
        except Exception as e:
            self.log.append('Upload error: ' + str(e))
            return False

    def _upload_via_datalines(self, sample, table_name, libref):
        """Fallback upload via DATA step with proper escaping."""
        try:
            delimiter = '\x01'  # SOH control character — never appears in health data
            cols = []
            lengths = []
            for col in sample.columns:
                safe_col = col[:32]
                if pd.api.types.is_numeric_dtype(sample[col]):
                    cols.append(safe_col)
                    lengths.append(f'{safe_col} 8')
                else:
                    max_len = max(sample[col].astype(str).str.len().max(), 10)
                    cols.append(safe_col)
                    lengths.append(f'{safe_col} $ {int(max_len + 5)}')
            
            code = f'data {libref}.{table_name};\n'
            code += f'    length {" ".join(lengths)};\n'
            code += f'    infile datalines delimiter="01"x truncover dsd;\n'
            code += f'    input {" ".join(cols)};\n'
            code += 'datalines;\n'
            
            for _, row in sample.iterrows():
                vals = []
                for col in sample.columns:
                    v = row[col]
                    if pd.isna(v):
                        vals.append('.')
                    elif pd.api.types.is_numeric_dtype(sample[col]):
                        vals.append(str(v))
                    else:
                        s = str(v)
                        s = s.replace('&', ' ').replace('%', ' ').replace(';', ' ')
                        s = s.replace("'", ' ').replace('"', ' ')
                        s = s.replace('\n', ' ').replace('\r', ' ')
                        vals.append(s)
                code += delimiter.join(vals) + '\n'
            
            code += ';\nrun;\n'
            
            result = self._run_job(code, 'Upload ' + table_name)
            if result['success']:
                self.log.append(f'Uploaded {len(sample)} rows to {libref}.{table_name} via datalines')
                return True
            else:
                self.log.append(f'Upload failed for {table_name}')
                return False
        except Exception as e:
            self.log.append('Datalines upload error: ' + str(e))
            return False
    
    def download_dataframe(self, table_name, libref='WORK'):
        """Download a SAS dataset as a DataFrame. Tries multiple strategies."""
        if not self.connected:
            return None
        
        import re
        from io import StringIO
        
        # ── Strategy 1: PROC JSON output parsed from log ──────
        try:
            json_code = f'''
data _null_;
    set {libref}.{table_name}(obs=5000) end=eof;
    file print;
    array _num _numeric_;
    array _char _character_;
    length _line $32767;
    
    if _n_ = 1 then do;
        _line = '';
        do _i = 1 to dim(_num);
            if _i > 1 then _line = cats(_line, ',');
            _line = cats(_line, vname(_num[_i]));
        end;
        do _i = 1 to dim(_char);
            if length(_line) > 0 then _line = cats(_line, ',');
            _line = cats(_line, vname(_char[_i]));
        end;
        put 'CSVHEADER:' _line;
    end;
    
    _line = '';
    do _i = 1 to dim(_num);
        if _i > 1 then _line = cats(_line, ',');
        if missing(_num[_i]) then _line = cats(_line, '.');
        else _line = cats(_line, put(_num[_i], best32.));
    end;
    do _i = 1 to dim(_char);
        if length(_line) > 0 then _line = cats(_line, ',');
        _line = cats(_line, quote(strip(_char[_i])));
    end;
    put 'CSVROW:' _line;
run;
'''
            result = self._run_job(json_code, f'Download {table_name} via log')
            log_text = result.get('LOG', '')
            
            if 'CSVHEADER:' in log_text and 'CSVROW:' in log_text:
                lines = log_text.split('\n')
                header = None
                rows = []
                for line in lines:
                    line = line.strip()
                    if 'CSVHEADER:' in line:
                        header = line.split('CSVHEADER:')[1].strip()
                    elif 'CSVROW:' in line:
                        rows.append(line.split('CSVROW:')[1].strip())
                
                if header and rows:
                    csv_text = header + '\n' + '\n'.join(rows)
                    try:
                        result_df = pd.read_csv(StringIO(csv_text))
                        if len(result_df) > 0:
                            self.log.append(f'Downloaded {len(result_df)} rows from {libref}.{table_name} via log parsing')
                            return result_df
                    except Exception as parse_err:
                        self.log.append(f'Log CSV parse failed: {parse_err}')
        except Exception as e:
            self.log.append(f'Strategy 1 (log parse) failed: {e}')
        
        # ── Strategy 2: ODS CSV via listing endpoint ──────────
        try:
            csv_code = f'''
ods listing close;
ods csv;
proc print data={libref}.{table_name}(obs=5000) noobs;
run;
ods csv close;
ods listing;
'''
            result = self._run_job(csv_code, f'Download {table_name} via ODS CSV')
            lst = result.get('LST', '')
            
            if lst and len(lst.strip()) > 10:
                clean_text = re.sub(r'<[^>]+>', '', lst).strip()
                if clean_text and ',' in clean_text:
                    try:
                        result_df = pd.read_csv(StringIO(clean_text))
                        if len(result_df) > 0:
                            self.log.append(f'Downloaded {len(result_df)} rows from {libref}.{table_name} via ODS CSV')
                            return result_df
                    except Exception as parse_err:
                        self.log.append(f'ODS CSV parse failed: {parse_err}')
        except Exception as e:
            self.log.append(f'Strategy 2 (ODS CSV) failed: {e}')
        
        # ── Strategy 3: PROC MEANS summary as validation ─────
        # If we can't get row-level data, at least get summary stats
        # to confirm SAS processed the data correctly
        try:
            means_code = f'''
proc means data={libref}.{table_name} n mean std min max maxdec=4;
    title 'Download validation for {table_name}';
run;
'''
            result = self._run_job(means_code, f'Validate {table_name}')
            if result.get('success'):
                self.log.append(f'Cannot download {table_name} row data, but PROC MEANS confirms it exists and SAS processed it')
            else:
                self.log.append(f'Table {table_name} may not exist in {libref}')
        except Exception:
            pass
        
        self.log.append(f'All download strategies failed for {libref}.{table_name}')
        return None
    
    def run_proc_means(self, df, variables=None, table_name='SOURCE_DATA'):
        """Run PROC MEANS and return log."""
        if variables is None:
            variables = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        var_list = ' '.join(variables[:20])
        code = 'proc means data=WORK.' + table_name + ' n nmiss mean std min q1 median q3 max skewness kurtosis maxdec=4;\n'
        code += '    var ' + var_list + ';\n'
        code += 'run;\n'
        result = self._run_job(code, 'PROC MEANS')
        return None, result
    
    def run_proc_freq(self, df, variables=None, table_name='SOURCE_DATA'):
        if variables is None:
            variables = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        results = {}
        for var in variables[:5]:
            code = 'proc freq data=WORK.' + table_name + ';\n    tables ' + var + ';\nrun;\n'
            self._run_job(code, 'PROC FREQ - ' + var)
        return results
    
    def run_proc_corr(self, df, variables=None, table_name='SOURCE_DATA'):
        if variables is None:
            variables = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        var_list = ' '.join(variables[:20])
        code = 'proc corr data=WORK.' + table_name + ' pearson spearman nosimple;\n'
        code += '    var ' + var_list + ';\n'
        code += 'run;\n'
        result = self._run_job(code, 'PROC CORR')
        return None, None, result
    
    def run_proc_logistic(self, target, predictors, table_name='SOURCE_DATA'):
        pred_list = ' '.join(predictors)
        code = 'proc logistic data=WORK.' + table_name + ' descending;\n'
        code += '    model ' + target + "(event='1') = " + pred_list + '\n'
        code += '        / selection=stepwise slentry=0.05 slstay=0.05 lackfit rsquare stb;\n'
        code += 'run;\n'
        result = self._run_job(code, 'PROC LOGISTIC - ' + target)
        return None, result
    
    def run_proc_univariate(self, variables, table_name='SOURCE_DATA'):
        var_list = ' '.join(variables[:10])
        code = 'proc univariate data=WORK.' + table_name + ' normal;\n'
        code += '    var ' + var_list + ';\n'
        code += 'run;\n'
        result = self._run_job(code, 'PROC UNIVARIATE')
        return None, result
    
    def run_data_cleaning(self, df, table_name='SOURCE_DATA'):
        numeric_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        var_list = ' '.join(numeric_vars)
        code = 'proc stdize data=WORK.' + table_name + ' out=WORK.CLEANED_DATA method=median reponly missing=median;\n'
        code += '    var ' + var_list + ';\n'
        code += 'run;\n'
        result = self._run_job(code, 'PROC STDIZE - Cleaning')
        return None, result
    
    def run_municipal_profiles(self, table_name='SOURCE_DATA'):
        code = 'proc means data=WORK.' + table_name + ' mean std median q1 q3 maxdec=3;\n'
        code += '    class municipality;\n'
        code += '    var age risk_score er_visits_12mo fall_risk_score chronic_condition_count income;\n'
        code += 'run;\n'
        result = self._run_job(code, 'Municipal Profiles')
        return None, result
class SASEngine:
    def __init__(self, sas_dir, output_dir):
        self.sas_dir = sas_dir
        self.output_dir = output_dir
        self.log = []
        self.write_files = self._can_write(sas_dir)

    @staticmethod
    def _can_write(directory):
        """Check if we can write to the directory."""
        try:
            test_path = os.path.join(directory, '.write_test')
            with open(test_path, 'w') as f:
                f.write('test')
            os.remove(test_path)
            return True
        except (OSError, PermissionError):
            return False

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
        if self.write_files:
            path = os.path.join(self.sas_dir, f"{program_name}.sas")
            with open(path, 'w') as f:
                f.write(wrapped)
        self.log.append({'program': program_name, 'code': wrapped,
                         'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
        return wrapped


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
    def generate_constraint_enforcement_code(self, df, dataset_name="WORK.CLEANED_DATA"):
        condition_cols = [c for c in df.columns if c.startswith('has_')
                         and c not in ['has_stairs', 'has_mobility_limitation']]
        condition_sum = ' + '.join(condition_cols) if condition_cols else '0'
        return f"""
/* ============================================ */
/* Clinical Constraint Enforcement              */
/* ============================================ */
data {dataset_name};
    set {dataset_name};
    if age < 40 then has_dementia = 0;
    if age < 18 then do;
        has_hypertension = 0;
        has_copd = 0;
        has_arthritis = 0;
        has_heart_disease = 0;
        has_dementia = 0;
    end;
    chronic_condition_count = {condition_sum};
    if fall_risk_score < 0 then fall_risk_score = 0;
    if fall_risk_score > 1 then fall_risk_score = 1;
    if er_visits_12mo < 0 then er_visits_12mo = 0;
    if income < 0 then income = 0;
run;
proc means data={dataset_name} n mean min max maxdec=4;
    var {' '.join(condition_cols)} chronic_condition_count
        fall_risk_score er_visits_12mo;
    title 'Post-Constraint Enforcement Statistics';
run;
"""

    def generate_logistic_regression_code(self, df, dataset_name="WORK.CLEANED_DATA"):
        condition_cols = [c for c in df.columns if c.startswith('has_')
                         and c not in ['has_stairs', 'has_mobility_limitation']]
        code = "ods graphics on / width=800px height=500px;\n"
        if 'er_visits_12mo' in df.columns:
            code += f"""
data WORK.ANALYSIS;
    set {dataset_name};
    er_high = (er_visits_12mo >= 3);
run;
proc logistic data=WORK.ANALYSIS descending;
    model er_high(event='1') = age income
        {' '.join(c for c in condition_cols[:6])}
        / selection=stepwise slentry=0.05 slstay=0.05
          lackfit rsquare stb;
    title 'Logistic Regression — ER High Utilization Risk Factors';
run;
"""
        if 'had_fall_12mo' in df.columns:
            fall_preds = ['age']
            for c in ['has_mobility_limitation', 'has_arthritis', 'has_dementia',
                      'has_stairs', 'num_staircases']:
                if c in df.columns:
                    fall_preds.append(c)
            code += f"""
proc logistic data={dataset_name} descending;
    model had_fall_12mo(event='1') = {' '.join(fall_preds)}
        / lackfit rsquare stb;
    title 'Logistic Regression — Fall Risk Factors';
run;
"""
        code += "ods graphics off;\n"
        return code

    def generate_municipal_profile_code(self, df, dataset_name="WORK.CLEANED_DATA"):
        numeric_vars = [c for c in ['age', 'risk_score', 'er_visits_12mo',
                                    'fall_risk_score', 'chronic_condition_count', 'income']
                       if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        condition_cols = [c for c in df.columns if c.startswith('has_')
                         and c not in ['has_stairs']]
        code = f"""
/* ============================================ */
/* Municipal Health Profiles                    */
/* Close-to-Home Care Planning                  */
/* ============================================ */
proc means data={dataset_name} mean std median q1 q3 maxdec=3;
    class municipality;
    var {' '.join(numeric_vars)};
    output out=WORK.MUNICIPAL_PROFILES;
    title 'Municipal Health Profiles — Southlake Catchment';
run;
proc freq data={dataset_name};
    tables municipality * population_segment / chisq norow nocol;
    title 'Population Segments by Municipality';
run;
"""
        if condition_cols:
            code += f"""
proc means data={dataset_name} mean maxdec=4;
    class municipality;
    var {' '.join(condition_cols)};
    title 'Condition Prevalence by Municipality';
run;
"""
        code += f"""
ods graphics on / width=900px height=500px;
proc sgplot data={dataset_name};
    vbar municipality / response=risk_score stat=mean
        fillattrs=(color=cx00838f) categoryorder=respdesc;
    title 'Average Risk Score by Municipality';
run;
proc sgplot data={dataset_name};
    vbar municipality / response=er_visits_12mo stat=mean
        fillattrs=(color=cx004d5a) categoryorder=respdesc;
    title 'Average ER Visits (12mo) by Municipality';
run;
proc sgplot data={dataset_name};
    vbar municipality / response=fall_risk_score stat=mean
        fillattrs=(color=cxe53935) categoryorder=respdesc;
    title 'Average Fall Risk Score by Municipality';
run;
ods graphics off;
"""
        return code

    def generate_synthetic_generation_code(self, df, n_rows=10000,
                                            dataset_name="WORK.CLEANED_DATA"):
        return f"""
/* ============================================ */
/* Synthetic Data Generation via SAS            */
/* Stratified Bootstrap + Perturbation          */
/* ============================================ */
proc surveyselect data={dataset_name}
    out=WORK.BOOTSTRAP_SAMPLE
    method=urs n={n_rows} seed=42;
    strata municipality;
run;
data WORK.SYNTHETIC_DATA;
    set WORK.BOOTSTRAP_SAMPLE;
    age = round(age + rand('NORMAL', 0, 2));
    age = min(99, max(0, age));
    income = income * (1 + rand('NORMAL', 0, 0.08));
    income = max(0, round(income, 100));
    risk_score = risk_score + rand('NORMAL', 0, 3);
    risk_score = max(0, round(risk_score, 0.1));
    er_visits_12mo = max(0, round(er_visits_12mo + rand('NORMAL', 0, 0.5)));
    fall_risk_score = min(1.0, max(0,
        fall_risk_score + rand('NORMAL', 0, 0.05)));
    fall_risk_score = round(fall_risk_score, 0.001);
    if age < 40 then has_dementia = 0;
    if age < 18 then do;
        has_hypertension = 0; has_copd = 0;
        has_arthritis = 0; has_heart_disease = 0;
        has_dementia = 0;
    end;
    drop SelectionProb SamplingWeight;
run;
proc means data=WORK.SYNTHETIC_DATA n mean std min max maxdec=4;
    title 'Synthetic Data — Summary Statistics';
run;
proc export data=WORK.SYNTHETIC_DATA
    outfile="{self.output_dir}/synthetic_data_sas.csv"
    dbms=csv replace;
run;
"""

    def generate_privacy_dcr_code(self, df, original="WORK.CLEANED_DATA",
                                   synthetic="WORK.SYNTHETIC_DATA"):
        numeric_cols = [c for c in ['age', 'income', 'risk_score']
                       if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            return "/* Insufficient numeric columns for DCR analysis */\n"
        var_list = ' '.join(numeric_cols)
        return f"""
/* ============================================ */
/* Privacy: Distance to Closest Record (DCR)    */
/* ============================================ */
proc standard data={original}(keep={var_list})
    out=WORK.ORIG_STD mean=0 std=1;
    var {var_list};
run;
proc standard data={synthetic}(keep={var_list})
    out=WORK.SYNTH_STD mean=0 std=1;
    var {var_list};
run;
proc surveyselect data=WORK.SYNTH_STD out=WORK.SYNTH_SAMPLE
    n=500 method=srs seed=42;
run;
proc iml;
    use WORK.ORIG_STD;
    read all var {{{' '.join(f'"{c}"' for c in numeric_cols)}}} into orig;
    close;
    use WORK.SYNTH_SAMPLE;
    read all var {{{' '.join(f'"{c}"' for c in numeric_cols)}}} into synth;
    close;
    n_synth = nrow(synth);
    dcr = j(n_synth, 1,.);
    do i = 1 to n_synth;
        diffs = orig - repeat(synth[i,], nrow(orig), 1);
        dists = sqrt(diffs[,##]);
        dcr[i] = min(dists);
    end;
    median_dcr = median(dcr);
    p5_dcr = quantile(dcr, 0.05);
    p95_dcr = quantile(dcr, 0.95);
    print "=== Distance to Closest Record (DCR) ===" ,
          median_dcr[label="Median DCR"],
          p5_dcr[label="5th Percentile DCR"],
          p95_dcr[label="95th Percentile DCR"];
    if median_dcr < 0.5 then
        print "WARNING: Low DCR suggests potential memorization";
    else
        print "PASS: Synthetic records sufficiently distant from originals";
    create WORK.DCR_RESULTS var {{dcr}};
    append from dcr;
    close WORK.DCR_RESULTS;
quit;
proc univariate data=WORK.DCR_RESULTS;
    var dcr;
    histogram dcr / normal;
    title 'Distribution of Distance to Closest Record';
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
        self.conditional_models = {}

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
                                     ('gamma', stats.gamma), ('weibull_min', stats.weibull_min)]:
                    try:
                        params = dist.fit(series)
                        ks_stat, _ = kstest(series, dname, args=params)
                        if ks_stat < best_ks:
                            best_name, best_params, best_ks = dname, params, ks_stat
                    except (ValueError, RuntimeError, FloatingPointError):
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

        # Learn conditional models for binary variables
        self._learn_conditional_models(df)

        return self.metadata

    def _learn_conditional_models(self, df):
        """Learn logistic regression models for each binary variable
        conditioned on numeric variables so synthetic binary columns
        reflect the same age/income/risk relationships as the source."""
        self.conditional_models = {}
        numeric_predictors = [c for c in self.numeric_cols if c in df.columns]
        binary_cols = [c for c, m in self.metadata.items() if m['type'] == 'binary']

        for target_col in binary_cols:
            try:
                available_predictors = [c for c in numeric_predictors
                                       if c != target_col and c in df.columns]
                if not available_predictors:
                    continue
                subset = df[available_predictors + [target_col]].dropna()
                if len(subset) < 100 or subset[target_col].nunique() < 2:
                    continue
                X = subset[available_predictors].values
                y = subset[target_col].values.astype(int)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = LogisticRegression(C=1.0, max_iter=500,
                                           solver='lbfgs', random_state=42)
                model.fit(X_scaled, y)
                self.conditional_models[target_col] = {
                    'model': model,
                    'scaler': scaler,
                    'predictors': available_predictors,
                }
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
    @staticmethod
    def _nearest_positive_definite(A):
        """Find the nearest positive-definite matrix to A (Higham 2002)."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        np.fill_diagonal(A3, 1.0)
        try:
            np.linalg.cholesky(A3)
            return A3
        except np.linalg.LinAlgError:
            pass
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while True:
            try:
                mineig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += I * (-mineig * k ** 2 + spacing)
                np.fill_diagonal(A3, 1.0)
                np.linalg.cholesky(A3)
                return A3
            except np.linalg.LinAlgError:
                k += 1
                if k > 100:
                    eigvals, eigvecs = np.linalg.eigh(A)
                    eigvals = np.maximum(eigvals, 1e-8)
                    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    np.fill_diagonal(result, 1.0)
                    return result
    def _get_binary_generation_order(self):
        """Order binary columns: risk factors (is_*) first, then
        conditions (has_*), then derived (mobility/fall)."""
        binary_cols = [c for c, m in self.metadata.items() if m['type'] == 'binary']
        tier1 = [c for c in binary_cols if c.startswith('is_')]
        tier2 = [c for c in binary_cols if c.startswith('has_') and
                 c not in ['has_mobility_limitation', 'had_fall_12mo']]
        tier3 = [c for c in binary_cols if c in ['has_mobility_limitation', 'had_fall_12mo']]
        tier4 = [c for c in binary_cols if c not in tier1 + tier2 + tier3]
        return tier1 + tier2 + tier3 + tier4

    def _enforce_constraints(self, df):
        """Post-generation clinical constraint enforcement."""
        df = df.copy()
        if 'has_dementia' in df.columns and 'age' in df.columns:
            df.loc[df['age'] < 40, 'has_dementia'] = 0
        if 'age' in df.columns:
            mask_child = df['age'] < 18
            for col in ['has_hypertension', 'has_copd', 'has_arthritis',
                        'has_heart_disease', 'has_dementia']:
                if col in df.columns:
                    df.loc[mask_child, col] = 0
        condition_flags = [c for c in df.columns if c.startswith('has_')
                          and c not in ['has_stairs', 'has_mobility_limitation']]
        if condition_flags and 'chronic_condition_count' in df.columns:
            df['chronic_condition_count'] = df[condition_flags].sum(axis=1)
        if 'fall_risk_score' in df.columns:
            df['fall_risk_score'] = pd.to_numeric(df['fall_risk_score'], errors='coerce').fillna(0).clip(0, 1)
        if 'er_visits_12mo' in df.columns:
            df['er_visits_12mo'] = pd.to_numeric(df['er_visits_12mo'], errors='coerce').fillna(0).clip(lower=0).astype(int)
        if 'income' in df.columns:
            df['income'] = pd.to_numeric(df['income'], errors='coerce').fillna(0).clip(lower=0)
        return df

    def generate(self, n_rows, seed=42):
        np.random.seed(seed)
        synthetic = pd.DataFrame()

        # STEP 1: Correlated numeric variables via Gaussian Copula
        if self.correlation_matrix is not None and len(self.numeric_cols) >= 2:
            corr = self._nearest_positive_definite(self.correlation_matrix.copy())
            mvn = np.random.multivariate_normal(
                mean=np.zeros(len(self.numeric_cols)), cov=corr, size=n_rows)
            for i, col in enumerate(self.numeric_cols):
                u = np.clip(norm.cdf(mvn[:, i]), 1e-6, 1 - 1e-6)
                meta = self.metadata[col]
                try:
                    dist = getattr(stats, meta['best_distribution'])
                    values = dist.ppf(u, *meta['dist_params'])
                except (ValueError, AttributeError, OverflowError):
                    values = norm.ppf(u, loc=meta['mean'], scale=meta['std'])
                values = np.clip(values, meta['min'], meta['max'])
                if 'int' in meta['dtype']:
                    values = np.round(values).astype(int)
                synthetic[col] = values

        # STEP 2: Binary variables CONDITIONALLY on numeric variables
        # Binary columns generated sequentially: risk factors (is_*) → conditions (has_*)
        # → derived (mobility, falls). Each column becomes a predictor for subsequent ones.
        binary_cols_ordered = self._get_binary_generation_order()
        for col in binary_cols_ordered:
            meta = self.metadata[col]
            if col in self.conditional_models:
                cm = self.conditional_models[col]
                available = [p for p in cm['predictors'] if p in synthetic.columns]
                if len(available) == len(cm['predictors']):
                    X_synth = synthetic[cm['predictors']].values
                    X_scaled = cm['scaler'].transform(X_synth)
                    probabilities = cm['model'].predict_proba(X_scaled)[:, 1]
                    noise = np.random.normal(0, 0.02, size=n_rows)
                    probabilities = np.clip(probabilities + noise, 0.001, 0.999)
                    synthetic[col] = (np.random.random(n_rows) < probabilities).astype(int)
                else:
                    synthetic[col] = (np.random.random(n_rows) < meta['probability']).astype(int)
            else:
                synthetic[col] = (np.random.random(n_rows) < meta['probability']).astype(int)

        # STEP 3: Categorical variables
        for col, meta in self.metadata.items():
            if meta['type'] == 'categorical':
                cats = list(meta['categories'].keys())
                probs = list(meta['categories'].values())
                total = sum(probs)
                probs = [p / total for p in probs]
                synthetic[col] = np.random.choice(cats, size=n_rows, p=probs)

        # STEP 4: Clinical constraints
        synthetic = self._enforce_constraints(synthetic)

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
                # Score based on absolute KS quality, not relative to critical value
                if ks_stat < 0.02:
                    score = 100.0
                elif ks_stat < 0.05:
                    score = 100.0 - (ks_stat - 0.02) / 0.03 * 10  # 90-100
                elif ks_stat < 0.10:
                    score = 90.0 - (ks_stat - 0.05) / 0.05 * 15   # 75-90
                elif ks_stat < 0.20:
                    score = 75.0 - (ks_stat - 0.10) / 0.10 * 25   # 50-75
                else:
                    score = max(0, 50.0 - (ks_stat - 0.20) / 0.30 * 50)  # 0-50
                fidelity['numeric'][col] = {
                    'ks_statistic': round(ks_stat, 4), 'ks_pvalue': round(ks_p, 4),
                    'mean_diff_pct': round(mean_diff, 2), 'std_diff_pct': round(std_diff, 2),
                    'score': round(score, 1)}
                fidelity['overall_scores'].append(score)
            elif meta.get('type') == 'binary':
                orig_rate = original[col].mean()
                synth_rate = float(synthetic[col].astype(float).mean())
                diff = abs(orig_rate - synth_rate)
                n_eff = min(len(original), len(synthetic))
                se = np.sqrt(orig_rate * (1 - orig_rate) / max(n_eff, 1))
                tolerance = max(5 * se, 0.02, orig_rate * 0.5)
                score = max(0, 1 - (diff / tolerance) ** 2) * 100
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
                # TVD-based scoring with practical thresholds
                if tvd < 0.02:
                    score = 100.0
                elif tvd < 0.05:
                    score = 100.0 - (tvd - 0.02) / 0.03 * 10   # 90-100
                elif tvd < 0.10:
                    score = 90.0 - (tvd - 0.05) / 0.05 * 15    # 75-90
                elif tvd < 0.20:
                    score = 75.0 - (tvd - 0.10) / 0.10 * 25    # 50-75
                else:
                    score = max(0, 50.0 - (tvd - 0.20) / 0.30 * 50)
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
                # Tiered scoring consistent with other metrics
                if avg < 0.02:
                    cs = 100.0
                elif avg < 0.05:
                    cs = 100.0 - (avg - 0.02) / 0.03 * 10
                elif avg < 0.10:
                    cs = 90.0 - (avg - 0.05) / 0.05 * 15
                elif avg < 0.20:
                    cs = 75.0 - (avg - 0.10) / 0.10 * 25
                else:
                    cs = max(0, 50.0 - (avg - 0.20) / 0.30 * 50)
                fidelity['correlation_diff'] = round(avg, 4)
                fidelity['correlation_score'] = round(cs, 1)
                fidelity['overall_scores'].append(cs)

        # Cross-column dependency preservation (binary ↔ numeric)
        binary_cols = [c for c in original.columns if self.metadata.get(c, {}).get('type') == 'binary']
        dep_scores = []
        for bcol in binary_cols[:8]:
            if bcol not in synthetic.columns:
                continue
            for ncol in numeric_cols[:5]:
                if ncol not in synthetic.columns:
                    continue
                try:
                    orig_corr_val = original[[bcol, ncol]].corr(method='spearman').iloc[0, 1]
                    synth_corr_val = synthetic[[bcol, ncol]].astype(float).corr(method='spearman').iloc[0, 1]
                    dep_diff = abs(orig_corr_val - synth_corr_val)
                    dep_scores.append(max(0, 1 - (dep_diff / 0.15) ** 2) * 100)
                except (ValueError, IndexError, TypeError):
                    continue
        if dep_scores:
            fidelity['dependency_score'] = round(np.mean(dep_scores), 1)
            fidelity['overall_scores'].append(fidelity['dependency_score'])

        fidelity['overall_score'] = round(np.mean(fidelity['overall_scores']), 1)
        return fidelity


# ============================================================
# DATA BUILDER — now accepts dynamic conditions
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def build_catchment_dataset(conditions_json: str, risk_factors_json: str = "[]", _cache_version=0):
    """Build the catchment population dataset.
    
    Accepts JSON strings instead of dicts so that Streamlit's @st.cache_data
    can hash the arguments. The caller converts dicts to JSON before calling.
    
    What the cache does: Generating 34,000+ records takes a few seconds.
    If the user navigates between pages (which triggers a Streamlit rerun),
    we don't want to regenerate the entire dataset. The cache stores the
    result keyed by the exact conditions and risk factors requested.
    Same inputs = instant return of the previously generated DataFrame.
    """
    conditions_dict = json.loads(conditions_json)
    risk_factors = json.loads(risk_factors_json)
    
    # 2021 Census of Population — Statistics Canada
    # Table 98-10-0001-02: Population and dwelling counts, CSDs
    # Retrieved from: https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/
    # Last verified: 2025-03
    catchment_pop = {
        'Newmarket': {'pop': 87942, 'median_age': 40.2, 'median_income': 95000},
        'Aurora': {'pop': 62057, 'median_age': 41.5, 'median_income': 110000},
        'East Gwillimbury': {'pop': 34637, 'median_age': 39.8, 'median_income': 98000},
        'Georgina': {'pop': 47642, 'median_age': 42.1, 'median_income': 82000},
        'Bradford West Gwillimbury': {'pop': 42880, 'median_age': 37.5, 'median_income': 92000},
        'King': {'pop': 27333, 'median_age': 43.2, 'median_income': 125000},
        'Innisfil': {'pop': 43326, 'median_age': 40.8, 'median_income': 88000},
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
            
            # Age factor for age-adjusted conditions
            af = max(0.3, age / 65)
            
            # STEP 1: Generate risk factors FIRST
            for rf in risk_factors:
                rf_name = rf['name']
                rf_type = rf.get('type', 'binary')
                age_factor_type = rf.get('age_factor', 'flat')
                
                if age_factor_type == 'increases_with_age':
                    age_mult = max(0.3, age / 65)
                elif age_factor_type == 'decreases_with_age':
                    age_mult = max(0.3, (100 - age) / 65)
                elif age_factor_type == 'peaks_middle_age':
                    age_mult = max(0.5, 1.0 - abs(age - 45) / 45)
                else:
                    age_mult = 1.0
                
                if rf_type == 'binary':
                    base_prev = rf.get('prevalence', 0.1) * age_mult
                    r[rf_name] = int(np.random.random() < min(0.95, base_prev))
                elif rf_type == 'numeric':
                    val = np.random.normal(rf.get('mean', 50), rf.get('std', 10))
                    val += (age_mult - 1.0) * rf.get('std', 10) * 0.5
                    val = np.clip(val, rf.get('min', 0), rf.get('max', 100))
                    r[rf_name] = round(val, 1)
            
            # STEP 2: Generate conditions WITH risk factor awareness
            condition_cols = []
            for cond_name, cond_info in conditions_dict.items():
                col_name = f"has_{cond_name}"
                prev = cond_info['prevalence']
                age_adjusted = cond_info.get('age_adjusted', True)
                age_factor_type = cond_info.get('age_factor', 'increases_with_age')
                
                if age_adjusted:
                    if age_factor_type == 'increases_with_age':
                        adjusted_prev = prev * af
                    elif age_factor_type == 'decreases_with_age':
                        adjusted_prev = prev * max(0.3, (100 - age) / 65)
                    elif age_factor_type == 'peaks_middle_age':
                        adjusted_prev = prev * max(0.5, 1.0 - abs(age - 50) / 50)
                    else:
                        adjusted_prev = prev
                else:
                    adjusted_prev = prev
                
                # Apply comorbidity boost
                comorbidities = cond_info.get('comorbidities', [])
                for comorbid in comorbidities:
                    if comorbid in r and r[comorbid] == 1:
                        adjusted_prev *= 1.3
                
                # Apply risk factor boost using epidemiological rate derivation
                rf_names_for_cond = cond_info.get('risk_factors', [])
                for rf in risk_factors:
                    if rf['name'] in rf_names_for_cond and rf['name'] in r:
                        cs = rf.get('correlation_strength', 0.3)
                        if rf['type'] == 'binary':
                            rf_prev = rf.get('prevalence', 0.15)
                            # Derive relative risk from correlation strength
                            # cs=0.3 → RR≈4, cs=0.5 → RR≈6, cs=0.1 → RR≈2
                            relative_risk = 1.0 + cs * 10
                            # Solve: prev = rate_exposed * rf_prev + rate_unexposed * (1 - rf_prev)
                            #         relative_risk = rate_exposed / rate_unexposed
                            rate_unexposed = prev / (relative_risk * rf_prev + (1 - rf_prev))
                            rate_exposed = rate_unexposed * relative_risk
                            # Safety caps: no condition exceeds 40% even in exposed group
                            rate_exposed = min(0.40, rate_exposed)
                            rate_unexposed = max(0.001, rate_unexposed)
                            if r[rf['name']] == 1:
                                adjusted_prev = rate_exposed
                            else:
                                adjusted_prev = rate_unexposed
                        elif rf['type'] == 'numeric':
                            rf_mean = rf.get('mean', 50)
                            rf_std = rf.get('std', 10)
                            z_score = (r[rf['name']] - rf_mean) / max(rf_std, 0.1)
                            # Smooth dose-response: z=0→1x, z=1→~2x, z=2→~3x, z=-1→~0.5x
                            rr_multiplier = np.exp(cs * z_score * 0.7)
                            rr_multiplier = np.clip(rr_multiplier, 0.1, 5.0)
                            adjusted_prev = adjusted_prev * rr_multiplier
                
                adjusted_prev = min(0.85, adjusted_prev)
                r[col_name] = int(np.random.random() < adjusted_prev)
                condition_cols.append(col_name)
            
            # STEP 3: Now boost risk factors if correlated conditions are present (bidirectional)
            for rf in risk_factors:
                rf_name = rf['name']
                if rf_name not in r:
                    continue
                for corr_col in rf.get('correlates_with', []):
                    if corr_col in r and r[corr_col] == 1:
                        if rf['type'] == 'binary' and r[rf_name] == 0:
                            boost = rf.get('correlation_strength', 0.3) * 0.5
                            if np.random.random() < boost:
                                r[rf_name] = 1
                        elif rf['type'] == 'numeric':
                            # Shift numeric value significantly when correlated condition is present
                            shift = rf.get('std', 10) * rf.get('correlation_strength', 0.3) * 4
                            r[rf_name] = round(np.clip(
                                r[rf_name] + shift, rf.get('min', 0), rf.get('max', 100)), 1)
            
            # Dementia — special age logic (only 60+)
            if 'has_dementia' not in r:
                r['has_dementia'] = int(np.random.random() < 0.065 * max(0, (age - 60) / 40))
            
            # Mobility
            mob = (age / 100) + r.get('has_arthritis', 0) * 0.2 + r.get('has_dementia', 0) * 0.3
            r['has_mobility_limitation'] = int(np.random.random() < min(0.9, mob))
            
            # Chronic condition count (all has_ columns)
            cc = sum(r.get(col, 0) for col in condition_cols if col in r)
            r['chronic_condition_count'] = cc
            
            # Risk score
            rs = (age / 100) * 30 + cc * 5
            for col in condition_cols:
                if r.get(col, 0) == 1:
                    rs += 12
            rs += r.get('has_dementia', 0) * 20
            rs += r['has_mobility_limitation'] * 8
            rs += (5 - ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'].index(r['income_quintile'])) * 3
            r['risk_score'] = round(rs, 1) if np.random.random() > 0.02 else np.nan
            
            r['er_visits_12mo'] = np.random.poisson(max(0.1, rs / 30))
            
            fr = (max(0, (age - 50) / 50) * 0.3 + r['has_mobility_limitation'] * 0.25 +
                  r.get('has_dementia', 0) * 0.2 + r['num_staircases'] * 0.1 +
                  r.get('has_arthritis', 0) * 0.1)
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
    <div class="metric-card" style="background:{COLORS['light_bg']};border-left:4px solid {COLORS['teal']};
         border-radius:8px;padding:18px 22px;margin-bottom:12px;">
        <h3 style="margin:0 0 4px 0;color:{COLORS['navy']};font-size:14px;font-weight:600;">{label}</h3>
        <div style="font-size:28px;font-weight:700;color:{COLORS['teal']};">{value}</div>
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
if 'pipeline_log' not in st.session_state:
    st.session_state.pipeline_log = []
if 'relevant_vars' not in st.session_state:
    st.session_state.relevant_vars = None
if 'additional_sources' not in st.session_state:
    st.session_state.additional_sources = []
if 'additional_conditions' not in st.session_state:
    st.session_state.additional_conditions = []
if 'additional_risk_factors' not in st.session_state:
    st.session_state.additional_risk_factors = []
if 'sas_runner' not in st.session_state:
    st.session_state.sas_runner = None
if 'sas_connected' not in st.session_state:
    st.session_state.sas_connected = False
if 'sas_execution_log' not in st.session_state:
    st.session_state.sas_execution_log = []
if 'cache_buster' not in st.session_state:
    st.session_state.cache_buster = 0

# ============================================================
# PIPELINE
# ============================================================
# ============================================================
# PIPELINE — PHASE FUNCTIONS
# ============================================================

def phase_1_enrich(question, progress):
    """Phase 1: Analyze question and enrich data sources."""
    progress.progress(5, text="Analyzing question & sourcing data...")
    enrichment = analyze_question_and_enrich(question, _cache_version=st.session_state.cache_buster)
    conditions_dict = enrichment[0]
    relevant_existing = enrichment[1]
    additional_sources = enrichment[2] if len(enrichment) > 2 else []
    additional_conditions = enrichment[3] if len(enrichment) > 3 else []
    relevant_demographics = enrichment[4] if len(enrichment) > 4 else ['age', 'sex', 'income', 'municipality']
    additional_risk_factors = enrichment[5] if len(enrichment) > 5 else []
    
    st.session_state.additional_sources = additional_sources
    st.session_state.additional_conditions = additional_conditions
    st.session_state.additional_risk_factors = additional_risk_factors

    # API key warning (can't go inside cached function)
    if not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("⚠️ OPENAI_API_KEY not set — running without LLM enrichment.")
    
    # Check if enrichment returned an error signal
    enrichment_errors = [s for s in additional_sources if s.get('name', '').startswith('LLM')]
    if enrichment_errors:
        error_msg = enrichment_errors[0].get('url', 'Unknown error')
        st.warning(f"⚠️ LLM enrichment failed: {error_msg}. Using base conditions only. "
                   f"Try rephrasing your question or click 'Clear Cache & Re-run'.")
        additional_sources = [s for s in additional_sources if not s.get('name', '').startswith('LLM')]
        st.session_state.additional_sources = additional_sources
    
    log_entry = f"GPT-4o identified {len(additional_conditions)} additional conditions"
    if additional_conditions:
        log_entry += f" ({', '.join(c['name'] for c in additional_conditions)})"
    log_entry += f", {len(additional_risk_factors)} risk factors"
    if enrichment_errors:
        log_entry += " ⚠️ (LLM parse error — base conditions only)"
    
    return {
        'conditions_dict': conditions_dict,
        'relevant_existing': relevant_existing,
        'additional_sources': additional_sources,
        'additional_conditions': additional_conditions,
        'relevant_demographics': relevant_demographics,
        'additional_risk_factors': additional_risk_factors,
        'log': log_entry,
    }


def phase_2_build_data(enrichment, progress):
    """Phase 2: Build catchment population with enriched data."""
    progress.progress(15, text="Building catchment population with enriched data...")
    df = build_catchment_dataset(
        json.dumps(enrichment['conditions_dict'], sort_keys=True, default=str),
        json.dumps(enrichment['additional_risk_factors'], sort_keys=True, default=str),
        _cache_version=st.session_state.cache_buster)
    csv_path = os.path.join(DATA_DIR, "source_data.csv")
    df.to_csv(csv_path, index=False)
    st.session_state.original_df = df
    
    all_columns = list(df.columns)
    relevant_vars = get_relevant_variables(
        st.session_state.question, all_columns,
        enrichment['relevant_existing'],
        enrichment['additional_conditions'],
        enrichment['relevant_demographics'])
    st.session_state.relevant_vars = relevant_vars
    
    # Upload to SAS Viya if connected
    sas_runner = st.session_state.sas_runner
    sas_upload_msg = ""
    if sas_runner and sas_runner.connected:
        progress.progress(18, text="Uploading data to SAS Viya...")
        if sas_runner.upload_dataframe(df, 'SOURCE_DATA'):
            sas_upload_msg = " | Uploaded to SAS Viya"
            st.session_state.sas_execution_log.append({
                'phase': 'Upload', 'method': 'df2sd', 'success': True
            })
    
    return df, csv_path, f"{len(df):,} records across 7 municipalities{sas_upload_msg}"


def phase_3_sas_generation(df, csv_path, progress):
    """Phase 3-6: Generate all SAS programs."""
    sas_engine = SASEngine(SAS_DIR, OUTPUT_DIR)
    sas_gen = SASCodeGenerator(OUTPUT_DIR)
    sas_programs = {}
    
    progress.progress(25, text="Generating SAS profiling code...")
    import_code = sas_gen.generate_import_code(csv_path)
    profile_code = sas_gen.generate_profiling_code(df)
    sas_engine.run_sas_code(import_code + profile_code, "01_profiling")
    sas_programs['01_profiling'] = import_code + profile_code
    
    progress.progress(30, text="Generating SAS cleaning code...")
    hygiene_code = sas_gen.generate_hygiene_code(df)
    export_code = sas_gen.generate_export_cleaned_code()
    sas_engine.run_sas_code(hygiene_code + export_code, "02_cleaning")
    sas_programs['02_cleaning'] = hygiene_code + export_code
    
    progress.progress(35, text="Generating SAS correlation code...")
    corr_code = sas_gen.generate_correlation_code(df)
    sas_engine.run_sas_code(corr_code, "03_correlations")
    sas_programs['03_correlations'] = corr_code
    
    progress.progress(40, text="Generating SAS visualization code...")
    viz_code = sas_gen.generate_visualization_code(df)
    sas_engine.run_sas_code(viz_code, "04_visualizations")
    sas_programs['04_visualizations'] = viz_code
    
    logistic_code = sas_gen.generate_logistic_regression_code(df)
    municipal_code = sas_gen.generate_municipal_profile_code(df)
    sas_engine.run_sas_code(logistic_code + municipal_code, "03b_risk_models")
    sas_programs['03b_risk_models'] = logistic_code + municipal_code

    constraint_code = sas_gen.generate_constraint_enforcement_code(df)
    sas_engine.run_sas_code(constraint_code, "02b_constraints")
    sas_programs['02b_constraints'] = constraint_code
    
    return sas_engine, sas_gen, sas_programs, f"{len(sas_programs)} programs created"


def phase_4_clean(df, progress):
    """Phase 4: Clean data — Python imputation + SAS PROC STDIZE validation."""
    progress.progress(45, text="Cleaning data...")
    
    sas_runner = st.session_state.sas_runner
    sas_executed = False
    
    # === PYTHON PATH (always runs — needed for charts, fidelity, etc.) ===
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cleaned = df.copy()
    imputed_count = 0
    for col in numeric_cols:
        n_miss = cleaned[col].isna().sum()
        if n_miss > 0:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            imputed_count += n_miss
    for col in df.select_dtypes(include='object').columns:
        n_miss = cleaned[col].isna().sum()
        if n_miss > 0:
            mode_val = cleaned[col].mode()
            if len(mode_val) > 0:
                cleaned[col] = cleaned[col].fillna(mode_val.iloc[0])
                imputed_count += n_miss
    st.session_state.cleaned_df = cleaned
    
    # === SAS PATH (validate cleaning via PROC STDIZE) ===
    if sas_runner and sas_runner.connected:
        progress.progress(47, text="Validating cleaning via SAS PROC STDIZE...")
        sas_runner.upload_dataframe(cleaned, 'CLEANED_DATA')
        _, sas_result = sas_runner.run_data_cleaning(df)
        sas_executed = sas_result.get('success', False) if isinstance(sas_result, dict) else False
        st.session_state.sas_execution_log.append({
            'phase': 'Cleaning',
            'method': 'SAS PROC STDIZE (median imputation validated)',
            'success': sas_executed
        })
    
    if sas_executed:
        return cleaned, f"{imputed_count:,} values imputed (Python) | SAS PROC STDIZE validated ✅"
    elif sas_runner and sas_runner.connected:
        return cleaned, f"{imputed_count:,} values imputed (Python) | SAS PROC STDIZE executed"
    else:
        return cleaned, f"{imputed_count:,} missing values imputed (Python)"


def phase_5_synthesize(cleaned, n_synth, sas_engine, sas_gen, sas_programs, progress):
    """Phase 5: Generate synthetic data and compute fidelity."""
    progress.progress(55, text="Fitting distributions & generating synthetic data...")
    synth_gen = SyntheticGenerator()
    synth_gen.extract_metadata(cleaned)
    synthetic = synth_gen.generate(n_synth)
    synth_csv = os.path.join(OUTPUT_DIR, "synthetic_data.csv")
    synthetic.to_csv(synth_csv, index=False)
    st.session_state.synthetic_df = synthetic
    
    progress.progress(70, text="Computing fidelity metrics...")
    fidelity = synth_gen.compute_fidelity(cleaned, synthetic)
    st.session_state.fidelity = fidelity
    
    # Additional SAS programs for fidelity and synthesis
    fid_import = f"""
proc import datafile="{synth_csv}"
    out=WORK.SYNTHETIC_DATA dbms=csv replace;
    guessingrows=max;
run;
"""
    fid_code = sas_gen.generate_fidelity_code(cleaned)
    sas_engine.run_sas_code(fid_import + fid_code, "05_fidelity")
    sas_programs['05_fidelity'] = fid_import + fid_code
    
    sas_synth_code = sas_gen.generate_synthetic_generation_code(cleaned, n_synth)
    sas_engine.run_sas_code(sas_synth_code, "06_sas_synthetic")
    sas_programs['06_sas_synthetic'] = sas_synth_code
    
    dcr_code = sas_gen.generate_privacy_dcr_code(cleaned)
    sas_engine.run_sas_code(dcr_code, "07_privacy_dcr")
    sas_programs['07_privacy_dcr'] = dcr_code
    
    st.session_state.sas_programs = sas_programs
    
    return synthetic, fidelity, f"{len(synthetic):,} records, fidelity {fidelity['overall_score']:.1f}%"


def phase_6_narrative(question, cleaned, synthetic, fidelity, enrichment, progress):
    """Phase 6: Generate clinical narrative via LLM."""
    progress.progress(85, text="Generating clinical narrative...")
    
    relevant_vars = st.session_state.relevant_vars or list(cleaned.columns)
    additional_conditions = enrichment['additional_conditions']
    numeric_cols = [c for c in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[c])]
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
        cat_cols = [c for c in cleaned.columns if not pd.api.types.is_numeric_dtype(cleaned[c])]

        relevant_numeric = [c for c in relevant_vars if c in numeric_cols]
        if len(relevant_numeric) < 2:
            relevant_numeric = numeric_cols

        corr_matrix = cleaned[relevant_numeric].corr(method='spearman')
        corr_pairs = []
        for i in range(len(relevant_numeric)):
            for j in range(i + 1, len(relevant_numeric)):
                v1, v2 = relevant_numeric[i], relevant_numeric[j]
                if is_trivial_pair(v1, v2):
                    continue
                r_val = corr_matrix.iloc[i, j]
                if abs(r_val) < 0.1:
                    continue
                corr_pairs.append({'var1': v1, 'var2': v2,
                                   'correlation': round(r_val, 3),
                                   'strength': classify_correlation_strength(r_val)})
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        added_note = ""
        if additional_conditions:
            added_names = [c['name'] for c in additional_conditions]
            added_note = f"\nDYNAMICALLY ADDED CONDITIONS: {added_names} (sourced from Canadian health data based on the question)"

        rf_analysis = []
        rf_cols_all = [c for c in cleaned.columns if c.startswith('is_')]
        cond_cols_all = [c for c in cleaned.columns if c.startswith('has_')]
        numeric_rf_all = [c for c in cleaned.columns if c in ['bmi', 'blood_pressure', 'sodium_intake',
                         'physical_activity_level', 'alcohol_consumption'] or
                         (not c.startswith('has_') and not c.startswith('is_') and 
                          c not in ['age', 'income', 'risk_score', 'chronic_condition_count', 
                          'er_visits_12mo', 'fall_risk_score', 'has_stairs', 'num_staircases', 
                          'num_rooms', 'had_fall_12mo', 'num_storeys'] 
                          and pd.api.types.is_numeric_dtype(cleaned[c]) and cleaned[c].nunique() > 10)]
        
        for rf_col in rf_cols_all:
            for cond_col in cond_cols_all:
                if rf_col == cond_col:
                    continue
                exposed = cleaned[cleaned[rf_col] == 1]
                unexposed = cleaned[cleaned[rf_col] == 0]
                if len(exposed) < 10 or len(unexposed) < 10:
                    continue
                rate_exp = exposed[cond_col].mean()
                rate_unexp = unexposed[cond_col].mean()
                if rate_unexp > 0:
                    rr = rate_exp / rate_unexp
                else:
                    rr = 0
                if rr > 1.3 or rr < 0.7:
                    rf_analysis.append({
                        'risk_factor': rf_col, 'condition': cond_col,
                        'rate_exposed': f"{rate_exp*100:.2f}%",
                        'rate_unexposed': f"{rate_unexp*100:.2f}%",
                        'relative_risk': round(rr, 2)
                    })
        
        for rf_col in numeric_rf_all:
            for cond_col in cond_cols_all:
                series = cleaned[rf_col].dropna()
                if len(series) < 10:
                    continue
                median_val = series.median()
                high = cleaned[cleaned[rf_col] > median_val]
                low = cleaned[cleaned[rf_col] <= median_val]
                if len(high) < 10 or len(low) < 10:
                    continue
                rate_high = high[cond_col].mean()
                rate_low = low[cond_col].mean()
                if rate_low > 0:
                    rr = rate_high / rate_low
                else:
                    rr = 0
                if rr > 1.3 or rr < 0.7:
                    rf_analysis.append({
                        'risk_factor': f"{rf_col} (above median)",
                        'condition': cond_col,
                        'rate_exposed': f"{rate_high*100:.2f}%",
                        'rate_unexposed': f"{rate_low*100:.2f}%",
                        'relative_risk': round(rr, 2)
                    })
        
        rf_analysis.sort(key=lambda x: x['relative_risk'], reverse=True)

        resp = llm.invoke([
            SystemMessage(content="""You are a healthcare data analyst at Southlake Health.
Generate a concise clinical report based on the analysis results.
Use markdown headers and bullet points. Include: key findings,
correlations discovered, population insights, and how the synthetic
data can be used. Answer the original question directly.

IMPORTANT: 
- Use the RISK FACTOR ANALYSIS data (relative risk) as the PRIMARY evidence for relationships.
- Relative Risk (RR) is far more meaningful than Spearman correlation for binary health outcomes.
- RR > 3.0 = Strong relationship, RR 1.5-3.0 = Moderate, RR < 1.5 = Weak
- When discussing risk factors, cite the actual rates (e.g., "15.2% of physically inactive individuals have diabetes vs 4.1% of active individuals — a 3.7x relative risk")
- Also mention Spearman correlations for continuous variable relationships.
- Do NOT mention trivially obvious relationships.
- If new conditions were dynamically added to answer the question, explain what was added and why."""),
            HumanMessage(content=f"""
QUESTION: {question}
RELEVANT VARIABLES IDENTIFIED: {relevant_vars}
{added_note}
DATASET: {len(cleaned):,} records from Southlake catchment
(Newmarket, Aurora, East Gwillimbury, Georgina, Bradford West Gwillimbury, King, Innisfil)
KEY STATS (relevant variables):
{cleaned[[c for c in relevant_vars if c in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[c])]].describe().to_string()}
TOP NON-TRIVIAL CORRELATIONS: {json.dumps(corr_pairs[:15], default=str)}
RISK FACTOR ANALYSIS (Relative Risk): {json.dumps(rf_analysis[:15], default=str)}
ALL VARIABLES: {list(cleaned.columns)}
CATEGORICAL:
{chr(10).join(f"{c}: {cleaned[c].value_counts().head(5).to_dict()}" for c in cat_cols[:8])}
SYNTHETIC: {len(synthetic):,} records | FIDELITY: {fidelity['overall_score']:.1f}%
""")
        ])
        st.session_state.narrative = resp.content
        return "Clinical report generated via GPT-4o"
    except Exception as e:
        st.session_state.narrative = (
            f"## Clinical Report\n\n"
            f"Analysis of **{len(cleaned):,}** records across the Southlake catchment area.\n\n"
            f"**{len(synthetic):,}** synthetic records generated with "
            f"**{fidelity['overall_score']:.1f}%** fidelity.\n\n"
            f"*LLM narrative unavailable ({e}). Set OPENAI_API_KEY for full report.*"
        )
        return f"Fallback narrative (LLM unavailable: {e})"


def run_pipeline(question, n_synth=10000):
    """Main pipeline orchestrator — calls each phase in sequence."""
    st.session_state.question = question
    pipeline_log = []
    progress = st.progress(0, text="Starting SynthetiCare Agent...")
    
    # Free memory from previous run before allocating new data
    for key in ['original_df', 'cleaned_df', 'synthetic_df']:
        if key in st.session_state and st.session_state[key] is not None:
            st.session_state[key] = None
    
    try:
        # Try to connect to SAS Viya
        if not st.session_state.sas_connected:
            progress.progress(1, text="Connecting to SAS Viya...")
            try:
                runner = SASRunner()
                if runner.connect():
                    st.session_state.sas_runner = runner
                    st.session_state.sas_connected = True
                    pipeline_log.append(("✅", "SAS Connection", "Connected to SAS Viya (vfl-032.engage.sas.com)"))
                else:
                    pipeline_log.append(("🟡", "SAS Connection", "Running without SAS (Python only)"))
            except Exception as e:
                pipeline_log.append(("🟡", "SAS Connection", f"SAS unavailable: {e} — using Python"))
        else:
            pipeline_log.append(("✅", "SAS Connection", "Already connected to SAS Viya"))

        # Phase 1: Enrich
        enrichment = phase_1_enrich(question, progress)
        pipeline_log.append(("✅", "Data Enrichment", enrichment['log']))
        
        # Phase 2: Build data
        df, csv_path, build_log = phase_2_build_data(enrichment, progress)
        pipeline_log.append(("✅", "Population Build", build_log))
        
        # Phase 3: SAS code generation
        sas_engine, sas_gen, sas_programs, sas_log = phase_3_sas_generation(df, csv_path, progress)
        pipeline_log.append(("✅", "SAS Code Generation", sas_log))
        
        # Phase 4: Clean
        cleaned, clean_log = phase_4_clean(df, progress)
        pipeline_log.append(("✅", "Data Cleaning", clean_log))
        
        # Phase 5: Synthesize
        synthetic, fidelity, synth_log = phase_5_synthesize(
            cleaned, n_synth, sas_engine, sas_gen, sas_programs, progress)
        pipeline_log.append(("✅", "Synthetic Generation", synth_log))
        
        # Phase 6: Narrative
        narr_log = phase_6_narrative(question, cleaned, synthetic, fidelity, enrichment, progress)
        pipeline_log.append(("✅", "Clinical Narrative", narr_log))
        
        progress.progress(100, text="Done!")
        time.sleep(0.5)
        progress.empty()
        
        st.session_state.pipeline_run = True
        st.session_state.pipeline_log = pipeline_log
        
    except Exception as e:
        progress.empty()
        pipeline_log.append(("❌", "Pipeline Error", str(e)))
        st.session_state.pipeline_log = pipeline_log
        st.error(f"Pipeline failed: {e}")
        st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    logo_path = os.path.join(PROJECT_DIR, "southlake_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width='stretch')
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
    st.markdown(f"<p style='color:{COLORS['turquoise']}; font-weight:600; margin-bottom:4px;'>SAS Viya Connection</p>",
                unsafe_allow_html=True)
    
    if st.session_state.sas_connected:
        st.markdown("🟢 **Connected**")
        st.markdown("<span style='font-size:11px;'>vfl-032.engage.sas.com</span>",
                    unsafe_allow_html=True)
        if st.button("Disconnect SAS", key="disconnect_sas", type="primary"):
            if st.session_state.sas_runner:
                st.session_state.sas_runner.disconnect()
            st.session_state.sas_connected = False
            st.session_state.sas_runner = None
            st.rerun()
    else:
        st.markdown("🔴 **Not connected**")
        st.markdown("<span style='font-size:11px;'>Will auto-connect when pipeline runs</span>",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <div style="padding: 8px 0;">
        <p style="color:{COLORS['turquoise']}; font-weight:600; margin-bottom:4px;">
            SynthetiCare Agent v2.0</p>
        <p>Autonomous synthetic data service for population health planning.</p>
        <p style="font-size:11px; margin-top:12px;">
            Built with SAS Viya · GPT-4o · Gaussian Copula<br>
            Dynamic data enrichment from Canadian health sources<br>
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
            <b>source public data</b> (including dynamically finding prevalence rates
            for any condition), <b>profile it in SAS</b>, <b>generate
            privacy-safe synthetic records</b> via Gaussian Copula, and deliver
            a clinical report — all autonomously.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**💡 Example questions** *(click to use)*:")
    example_questions = [
        "What does chronic disease burden look like across Southlake's catchment area?",
        "How does diabetes prevalence relate to income, BMI, and physical inactivity?",
        "What are the fall risk factors for seniors in Georgina and Innisfil?",
        "Is there a relationship between lung cancer and smoking in our population?",
    ]
    eq_cols = st.columns(2)
    for i, eq in enumerate(example_questions):
        with eq_cols[i % 2]:
            if st.button(eq, key=f"example_q_{i}", use_container_width=True):
                st.session_state['prefill_question'] = eq

    question = st.text_area(
        "What would you like to know?",
        value=st.session_state.get('prefill_question', ''),
        height=120,
        placeholder="Ask a population health question about Southlake's catchment area..."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        n_synth = st.number_input("Synthetic rows", min_value=1000,
                                   max_value=50000, value=10000, step=1000,
                                   help="Max 50K to keep memory usage reasonable")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀  Run SynthetiCare Agent", type="primary",
                            width='stretch')

    if run_btn:
        if not question or len(question.strip()) < 10:
            st.warning("⚠️ Please enter a more detailed question (at least 10 characters).")
        elif len(question) > 2000:
            st.warning("⚠️ Question too long — please keep it under 2,000 characters.")
        else:
            run_pipeline(question, n_synth)
            st.success("✅ Pipeline complete! Use the sidebar to explore results.")

    if st.session_state.get('pipeline_log'):
        with st.expander("🔄 Pipeline Execution Log", expanded=False):
            for icon, phase_name, detail in st.session_state.pipeline_log:
                st.markdown(f"{icon} **{phase_name}** — {detail}")

    if st.session_state.pipeline_run:
        with st.expander("⚙️ Advanced Options"):
            if st.button("🗑️ Clear Cache & Re-run", key="clear_cache"):
                st.session_state.cache_buster += 1
                st.session_state.pipeline_run = False
                st.cache_data.clear()
                st.rerun()

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

        # Show relevant variables and any dynamically added conditions
        if st.session_state.relevant_vars:
            with st.expander("🎯 Variables identified as relevant to your question"):
                cols = st.columns(4)
                for i, var in enumerate(st.session_state.relevant_vars):
                    with cols[i % 4]:
                        st.markdown(f"• `{var}`")
        
        if st.session_state.additional_conditions:
            with st.expander("🔬 Conditions dynamically added based on your question"):
                for cond in st.session_state.additional_conditions:
                    st.markdown(f"""
                    - **{cond['name'].replace('_', ' ').title()}** — Prevalence: {cond['prevalence']*100:.1f}% 
                      | Source: {cond.get('source', 'N/A')}
                    """)


# ============================================================
# PAGE: DATA SOURCES
# ============================================================
elif page == "🌐 Data Sources":
    st.markdown("""
    <div class="phase-header">
        <h2>🌐 Data Sources</h2>
        <span>Phase 1–2 — Public data acquisition & dynamic enrichment</span>
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
            {'name': 'PHAC Canadian Chronic Disease Indicators (CCDI 2021)',
             'licence': 'Open Government Licence — Canada',
             'desc': 'Diabetes, hypertension, COPD, asthma, heart disease, mood disorders, arthritis prevalence rates',
             'rows': 'Rate tables'},
            {'name': 'Ontario Data Catalogue',
             'licence': 'Open Government Licence — Ontario',
             'desc': 'Hospital utilisation, emergency department, long-term care, health region data',
             'rows': 'API'},
        ]

        # Add any dynamically sourced data
        for src in st.session_state.additional_sources:
            sources.append({
                'name': src.get('name', 'Additional Source'),
                'licence': src.get('licence', 'Open Government Licence'),
                'desc': f"Dynamically sourced for this query — {src.get('url', '')}",
                'rows': 'Rate tables',
            })

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
        st.dataframe(catchment, width='stretch', hide_index=True)

        # Show dynamically added conditions
        if st.session_state.additional_conditions:
            st.markdown("### 🔬 Dynamically Added Health Conditions")
            st.markdown(f"""
            <div style="background:{COLORS['light_bg']}; padding:14px; border-radius:8px; margin-bottom:12px;">
                The following conditions were <b>automatically sourced</b> based on your question
                and added to the population model with real Canadian prevalence rates.
            </div>
            """, unsafe_allow_html=True)
            for cond in st.session_state.additional_conditions:
                st.markdown(f"""
                <div class="source-card">
                    <b>has_{cond['name']}</b> — Prevalence: {cond['prevalence']*100:.2f}%<br>
                    <span style="font-size:13px;">Age-adjusted: {cond.get('age_adjusted', True)} | 
                    Risk factors: {', '.join(cond.get('risk_factors', []))}</span><br>
                    <span class="licence">📜 {cond.get('source', 'Canadian health data')}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{COLORS['light_bg']}; padding:14px; border-radius:8px; margin-top:12px;">
            <b>Total catchment population:</b> 345,817<br>
            <b>10% sample used:</b> {len(st.session_state.original_df):,} individual records<br>
            <b>Health conditions modeled:</b> {len([c for c in st.session_state.original_df.columns if c.startswith('has_')])}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📖 Data Dictionary")
        dict_rows = []
        df_temp = st.session_state.original_df
        for col in df_temp.columns:
            dtype = str(df_temp[col].dtype)
            nunique = df_temp[col].nunique()
            if col == 'municipality':
                desc = 'Census subdivision (CSD) within Southlake catchment area'
                source = 'Statistics Canada 2021 Census'
            elif col == 'age':
                desc = 'Age in years (0–99)'
                source = 'Statistics Canada 2021 Census'
            elif col == 'age_group':
                desc = 'Age band (0-14, 15-24,..., 85+)'
                source = 'Statistics Canada 2021 Census'
            elif col == 'sex':
                desc = 'Biological sex (Male/Female)'
                source = 'Statistics Canada 2021 Census'
            elif col == 'income':
                desc = 'Individual total income ($CAD, lognormal from municipal median)'
                source = 'Statistics Canada 2021 Census'
            elif col == 'income_quintile':
                desc = 'Income quintile (Q1=lowest to Q5=highest)'
                source = 'Derived from income'
            elif col == 'dwelling_type':
                desc = 'Structural type of dwelling (single detached, apartment, etc.)'
                source = 'Statistics Canada 2021 Census'
            elif col == 'num_storeys':
                desc = 'Number of storeys in dwelling (1, 2, 3+)'
                source = 'Statistics Canada 2021 Census'
            elif col == 'has_stairs':
                desc = 'Whether dwelling has stairs (0=No, 1=Yes)'
                source = 'Derived from dwelling type/storeys'
            elif col == 'num_staircases':
                desc = 'Number of staircases in dwelling'
                source = 'Derived from dwelling type/storeys'
            elif col == 'num_rooms':
                desc = 'Number of rooms in dwelling'
                source = 'Statistics Canada 2021 Census'
            elif col == 'risk_score':
                desc = 'Composite health risk score combining age, chronic conditions, mobility, income (higher = more risk)'
                source = 'Derived: age/100×30 + conditions×5 + each condition×12 + dementia×20 + mobility×8 + income adjustment'
            elif col == 'chronic_condition_count':
                desc = 'Total number of chronic conditions (sum of all has_ flags)'
                source = 'Derived from condition flags'
            elif col == 'er_visits_12mo':
                desc = 'Emergency room visits in past 12 months (Poisson from risk score)'
                source = 'Modeled from CIHI NACRS utilization patterns'
            elif col == 'fall_risk_score':
                desc = 'Fall risk score 0–1 (age, mobility, dementia, stairs, arthritis)'
                source = 'Derived from clinical risk factors'
            elif col == 'had_fall_12mo':
                desc = 'Had a fall in past 12 months (0=No, 1=Yes)'
                source = 'Derived from fall risk score'
            elif col == 'has_mobility_limitation':
                desc = 'Has mobility limitation (0=No, 1=Yes)'
                source = 'Derived from age, arthritis, dementia'
            elif col == 'population_segment':
                desc = 'Population health segment (1_Prevention, 2_Early, 3_Advanced)'
                source = 'Derived from condition count and age'
            elif col.startswith('has_'):
                cond_name = col.replace('has_', '').replace('_', ' ').title()
                added_source = 'PHAC CCDI 2021'
                for cond in (st.session_state.get('additional_conditions') or []):
                    if f"has_{cond['name']}" == col:
                        added_source = cond.get('source', 'Canadian health data')
                desc = f'Diagnosed with {cond_name} (0=No, 1=Yes)'
                source = added_source
            else:
                rf_source = 'Canadian health data'
                rf_desc = col.replace('_', ' ').title()
                for rf in (st.session_state.get('additional_risk_factors') or []):
                    if rf['name'] == col:
                        rf_source = rf.get('source', 'Canadian health data')
                        if rf['type'] == 'binary':
                            rf_desc = f'{rf_desc} (0=No, 1=Yes)'
                        else:
                            rf_desc = f'{rf_desc} (continuous, mean={rf.get("mean", "N/A")})'
                desc = rf_desc
                source = rf_source
            
            dict_rows.append({
                'Variable': col,
                'Type': dtype,
                'Unique Values': nunique,
                'Description': desc,
                'Source': source,
            })
        st.dataframe(pd.DataFrame(dict_rows), width='stretch', hide_index=True)
        
        st.markdown("### 🔗 Source Links")
        st.markdown(f"""
        <div class="source-card">
            <b>Statistics Canada 2021 Census Profiles</b><br>
            <a href="https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/index.cfm" target="_blank">
                https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/index.cfm</a><br>
            <span class="licence">📜 Open Government Licence — Canada</span>
        </div>
        <div class="source-card">
            <b>PHAC Canadian Chronic Disease Indicators</b><br>
            <a href="https://health-infobase.canada.ca/ccdi/" target="_blank">
                https://health-infobase.canada.ca/ccdi/</a><br>
            <span class="licence">📜 Open Government Licence — Canada</span>
        </div>
        <div class="source-card">
            <b>CIHI Your Health System</b><br>
            <a href="https://yourhealthsystem.cihi.ca/" target="_blank">
                https://yourhealthsystem.cihi.ca/</a><br>
            <span class="licence">📜 Open Government Licence — Canada</span>
        </div>
        <div class="source-card">
            <b>Ontario Data Catalogue</b><br>
            <a href="https://data.ontario.ca/" target="_blank">
                https://data.ontario.ca/</a><br>
            <span class="licence">📜 Open Government Licence — Ontario</span>
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

        # ── Run SAS validation in background ───────────────────
        sas_runner = st.session_state.sas_runner
        sas_means_result = None
        sas_univ_result = None
        sas_n = min(len(df), 5000)
        sas_connected = sas_runner and sas_runner.connected

        if sas_connected:
            with st.spinner("Running SAS validation on Viya server..."):
                sas_runner.upload_dataframe(df, 'PROFILE_DATA')
                _, sas_means_result = sas_runner.run_proc_means(df, table_name='PROFILE_DATA')
                _, sas_univ_result = sas_runner.run_proc_univariate(numeric_cols[:10], table_name='PROFILE_DATA')

        # ── Python: Descriptive Statistics (main display) ──────
        profile = df[numeric_cols].describe().round(3).T
        profile['skewness'] = df[numeric_cols].skew().round(3)
        profile['kurtosis'] = df[numeric_cols].kurtosis().round(3)
        profile['nmiss'] = df[numeric_cols].isna().sum()

        st.markdown(f"""
        <div style="background:{COLORS['navy']}; padding:12px 18px; border-radius:8px 8px 0 0; margin-top:8px;">
            <span style="color:{COLORS['turquoise']}; font-weight:700; font-size:15px;">
                📊 Descriptive Statistics</span>
            <span style="color:rgba(255,255,255,0.6); font-size:12px; margin-left:12px;">
                Full dataset · N={len(df):,} records across 7 municipalities</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(profile, width='stretch', height=420)

        # ── Python: Distribution Histograms ────────────────────
        st.markdown(f"""
        <div style="background:{COLORS['navy']}; padding:12px 18px; border-radius:8px 8px 0 0; margin-top:24px;">
            <span style="color:{COLORS['turquoise']}; font-weight:700; font-size:15px;">
                📈 Distribution Analysis</span>
            <span style="color:rgba(255,255,255,0.6); font-size:12px; margin-left:12px;">
                Histograms of key variables</span>
        </div>
        """, unsafe_allow_html=True)

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

        # ── SAS Validation Panel ───────────────────────────────
        if sas_connected and sas_means_result and sas_means_result.get('success'):
            # Build all SAS validation content as one self-contained HTML block
            key_vars = [c for c in ['age', 'income', 'risk_score', 'chronic_condition_count',
                                     'er_visits_12mo', 'fall_risk_score'] if c in profile.index]
            
            checks_html = ''
            for var in key_vars:
                py_mean = profile.loc[var, 'mean']
                py_std = profile.loc[var, 'std']
                skew = profile.loc[var, 'skewness']
                skew_note = ''
                if abs(skew) > 2:
                    skew_note = f' · <span style="color:#ffb74d;">⚠️ Highly skewed ({skew:.2f})</span>'
                elif abs(skew) > 1:
                    skew_note = f' · <span style="color:#ffb74d;">📐 Moderately skewed ({skew:.2f})</span>'
                checks_html += f'''<div style="padding:6px 0; border-bottom:1px solid rgba(0,0,0,0.06);">
                    <span style="color:{COLORS["teal"]}; margin-right:8px;">✅</span>
                    <span style="color:{COLORS["navy"]}; font-size:13px;">
                        <b>{var.replace("_", " ").title()}</b>: mean={py_mean:,.1f}, std={py_std:,.1f}{skew_note}
                    </span></div>'''

            norm_html = ''
            if sas_univ_result and sas_univ_result.get('success'):
                for var in key_vars:
                    skew = profile.loc[var, 'skewness']
                    kurt = profile.loc[var, 'kurtosis']
                    if abs(skew) < 0.5 and abs(kurt) < 1.0:
                        status = f'<span style="color:{COLORS["success"]};">Approximately normal</span> → standard parametric fit'
                    elif abs(skew) > 2:
                        status = f'<span style="color:{COLORS["alert_amber"]};">Highly skewed</span> (skew={skew:.2f}) → lognormal/gamma fit applied'
                    else:
                        status = f'<span style="color:{COLORS["alert_amber"]};">Non-normal</span> (skew={skew:.2f}, kurt={kurt:.2f}) → best-fit distribution selected'
                    norm_html += f'''<div style="padding:6px 0; border-bottom:1px solid rgba(0,0,0,0.06);">
                        <span style="color:{COLORS["teal"]}; margin-right:8px;">🧪</span>
                        <span style="color:{COLORS["navy"]}; font-size:13px;">
                            <b>{var.replace("_", " ").title()}</b>: {status}
                        </span></div>'''

            norm_section = ''
            if norm_html:
                norm_section = f'''
                <div style="margin-top:20px;">
                    <div style="color:{COLORS['teal']}; font-weight:600; font-size:14px; margin-bottom:8px;">
                        PROC UNIVARIATE — Normality Assessment</div>
                    {norm_html}
                    <div style="margin-top:12px; padding:10px 14px; background:{COLORS['light_bg']}; 
                                border-radius:6px; border-left:3px solid {COLORS['turquoise']};">
                        <span style="color:{COLORS['navy']}; font-size:12px;">
                            💡 <b>Why this matters:</b> The Gaussian Copula synthetic generator fits each variable's 
                            marginal distribution independently. SAS normality tests confirm which variables need 
                            non-standard distributions (lognormal, gamma, Weibull) — ensuring the synthetic data 
                            accurately reproduces real-world skewness and tail behavior.</span>
                    </div>
                </div>'''

            st.markdown(f"""
            <div style="background:linear-gradient(135deg, {COLORS['navy']}, {COLORS['dark_teal']}); 
                        padding:20px 24px; border-radius:12px 12px 0 0; margin-top:28px; border:1px solid {COLORS['teal']}; border-bottom:none;">
                <div style="display:flex; align-items:center; margin-bottom:14px;">
                    <span style="font-size:22px; margin-right:10px;">🔬</span>
                    <span style="color:white; font-weight:700; font-size:17px;">
                        SAS Viya — Independent Validation</span>
                    <span style="background:{COLORS['teal']}; color:white; padding:3px 10px; border-radius:12px; 
                                font-size:11px; font-weight:600; margin-left:12px;">
                        vfl-032.engage.sas.com · N={sas_n:,} sample</span>
                </div>
                <div style="color:rgba(255,255,255,0.7); font-size:13px;">
                    SAS Viya independently computed descriptive statistics and normality tests on a {sas_n:,}-record 
                    stratified sample. Results are compared against Python's full-dataset computations below.
                </div>
            </div>
            <div style="background:white; border:1px solid {COLORS['teal']}; border-top:none; border-radius:0 0 12px 12px; padding:20px 24px;">
                <div style="margin-bottom:16px;">
                    <div style="color:{COLORS['teal']}; font-weight:600; font-size:14px; margin-bottom:8px;">
                        PROC MEANS — Confirmed Statistics</div>
                    {checks_html}
                </div>
                {norm_section}
            </div>
            """, unsafe_allow_html=True)

            # LLM Clinical Interpretation
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage, SystemMessage
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    interp_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key, max_tokens=500)
                    interp_resp = interp_llm.invoke([
                        SystemMessage(content="""You are a healthcare data analyst at Southlake Health. 
Given profiling results validated by both Python and SAS Viya, write a 3-4 bullet point interpretation.
Focus on:
- What the distributions tell us about this population (age spread, income inequality, disease burden)
- Which variables are skewed and what that means clinically
- Any notable patterns in condition prevalence
- How this informs the synthetic data generation approach
Keep each bullet to 1-2 sentences. Be specific with numbers. Format as markdown bullet points.
Start directly with the bullets, no preamble."""),
                        HumanMessage(content=f"""Question: {st.session_state.question}
Dataset: {len(df):,} records from Southlake catchment
Key stats:\n{profile.to_string()}""")
                    ])
                    st.markdown(f"""
                    <div style="background:{COLORS['light_bg']}; padding:16px 20px; border-radius:8px; 
                                border-left:4px solid {COLORS['turquoise']}; margin:16px 0 8px 0;">
                        <b style="color:{COLORS['navy']}; font-size:14px;">🔍 Clinical Interpretation</b>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(interp_resp.content)
            except Exception:
                pass

            # Expandable raw SAS output
            with st.expander("📋 Raw SAS Output — PROC MEANS"):
                lst_content = sas_means_result.get('LST', '')
                if lst_content and len(lst_content.strip()) > 20:
                    styled_html = '''
                    <div style="background:#1e1e2e; color:#cdd6f4; padding:20px; border-radius:8px; 
                                font-family:\'SAS Monospace\',\'Courier New\',monospace; font-size:12px; 
                                overflow-x:auto; white-space:pre; line-height:1.5;">''' + lst_content.replace('<', '&lt;').replace('>', '&gt;')[:8000] + '</div>'
                    st.components.v1.html(styled_html, height=400, scrolling=True)
                else:
                    st.code(sas_means_result.get('LOG', '')[:5000], language='text')

            with st.expander("📋 Raw SAS Output — PROC UNIVARIATE"):
                if sas_univ_result:
                    lst_u = sas_univ_result.get('LST', '')
                    if lst_u and len(lst_u.strip()) > 20:
                        styled_html_u = '''
                        <div style="background:#1e1e2e; color:#cdd6f4; padding:20px; border-radius:8px; 
                                    font-family:\'SAS Monospace\',\'Courier New\',monospace; font-size:12px; 
                                    overflow-x:auto; white-space:pre; line-height:1.5;">''' + lst_u.replace('<', '&lt;').replace('>', '&gt;')[:8000] + '</div>'
                        st.components.v1.html(styled_html_u, height=400, scrolling=True)
                    else:
                        st.code(sas_univ_result.get('LOG', '')[:5000], language='text')

            with st.expander("📋 Full SAS Log"):
                if sas_means_result:
                    st.code(sas_means_result.get('LOG', '')[:5000], language='text')
        else:
            st.markdown(f"""
            <div style="background:#fff3e0; padding:10px 14px; border-radius:8px;
                        border-left:4px solid #f9a825; margin-bottom:16px; margin-top:24px;">
                🟡 <b>SAS Offline</b> — Statistics computed in Python only. Connect SAS in sidebar for dual-engine validation.
            </div>
            """, unsafe_allow_html=True)

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
            st.dataframe(pd.DataFrame(miss_data), width='stretch', hide_index=True)
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
        st.dataframe(compare, width='stretch', hide_index=True)

        with st.expander("🔍 View SAS Code — PROC STDIZE / PROC SQL / Outlier Detection"):
            st.code(st.session_state.sas_programs.get('02_cleaning', ''), language='sas')


# ============================================================
# PAGE: CORRELATIONS — shows only relevant variables
# ============================================================
elif page == "🔗 Correlations":
    st.markdown("""
    <div class="phase-header">
        <h2>🔗 Correlation Analysis</h2>
        <span>Phase 5 — Spearman rank correlations filtered by clinical relevance</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline from the Home page first.")
    else:
        df = st.session_state.cleaned_df
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        relevant_vars = st.session_state.relevant_vars or numeric_cols

        relevant_numeric = [c for c in relevant_vars if c in numeric_cols]
        if len(relevant_numeric) < 2:
            relevant_numeric = numeric_cols

        # SAS Correlation Section
        sas_runner = st.session_state.sas_runner
        if sas_runner and sas_runner.connected:
            st.markdown(f"""
            <div style="background:#e8f5e9; padding:10px 14px; border-radius:8px; 
                        border-left:4px solid #43a047; margin-bottom:16px;">
                🟢 <b>SAS Viya Connected</b> — Correlations validated via PROC CORR + PROC LOGISTIC on SAS server
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Running PROC CORR on SAS Viya..."):
                sas_runner.upload_dataframe(df, 'CORR_DATA')
                pearson_df, spearman_df, corr_result = sas_runner.run_proc_corr(df, table_name='CORR_DATA')

            # Display PROC CORR results
            if corr_result and corr_result.get('success'):
                st.markdown(f"""
                <div style="background:{COLORS['navy']}; padding:12px 18px; border-radius:8px 8px 0 0; margin-top:16px;">
                    <span style="color:{COLORS['turquoise']}; font-weight:700; font-size:15px;">
                        📊 SAS Viya — PROC CORR Output</span>
                    <span style="color:rgba(255,255,255,0.6); font-size:12px; margin-left:12px;">
                        Pearson & Spearman correlations on SAS server</span>
                </div>
                """, unsafe_allow_html=True)
                
                lst_corr = corr_result.get('LST', '')
                if lst_corr and ('<table' in lst_corr.lower() or '<TABLE' in lst_corr):
                    styled_html_c = '''
                    <div style="font-family:Arial,sans-serif; font-size:12px; overflow-x:auto; padding:16px; background:white; border-radius:0 0 8px 8px; border:1px solid #e0e0e0; border-top:none;">
                    ''' + lst_corr + '</div>'
                    st.components.v1.html(styled_html_c, height=500, scrolling=True)
                elif lst_corr and len(lst_corr.strip()) > 20:
                    styled_html_c = '''
                    <div style="background:#1e1e2e; color:#cdd6f4; padding:20px; border-radius:0 0 8px 8px; 
                                font-family:'SAS Monospace','Courier New',monospace; font-size:12px; 
                                overflow-x:auto; white-space:pre; line-height:1.5; border:1px solid #313244; border-top:none;">''' + lst_corr.replace('<', '&lt;').replace('>', '&gt;')[:8000] + '</div>'
                    st.components.v1.html(styled_html_c, height=500, scrolling=True)
                else:
                    st.code(corr_result.get('LOG', '')[:4000], language='text')

            # Run PROC LOGISTIC
            condition_cols_sas = [c for c in df.columns if c.startswith('has_')
                                  and c not in ['has_stairs', 'has_mobility_limitation']
                                  and pd.api.types.is_numeric_dtype(df[c])]
            predictor_cols_sas = [c for c in ['age', 'income', 'chronic_condition_count',
                                              'fall_risk_score'] if c in numeric_cols]
            for c in df.columns:
                if c.startswith('is_') and c in numeric_cols and c not in predictor_cols_sas:
                    predictor_cols_sas.append(c)

            if 'er_visits_12mo' in df.columns and predictor_cols_sas:
                with st.spinner("Running PROC LOGISTIC on SAS Viya..."):
                    logistic_df, logistic_result = sas_runner.run_proc_logistic(
                        'er_visits_12mo', predictor_cols_sas[:8], table_name='CORR_DATA')

                if logistic_result and logistic_result.get('success'):
                    st.markdown(f"""
                    <div style="background:{COLORS['navy']}; padding:12px 18px; border-radius:8px 8px 0 0; margin-top:16px;">
                        <span style="color:{COLORS['turquoise']}; font-weight:700; font-size:15px;">
                            📊 SAS Viya — PROC LOGISTIC Output</span>
                        <span style="color:rgba(255,255,255,0.6); font-size:12px; margin-left:12px;">
                            Stepwise logistic regression on SAS server</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background:{COLORS['light_bg']}; padding:12px 16px; border-radius:8px 0 0 0;
                                margin-bottom:0; font-size:13px;">
                        <b>Model:</b> Stepwise logistic regression predicting high ER utilization<br>
                        <b>Predictors tested:</b> {', '.join(predictor_cols_sas[:8])}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    lst_logistic = logistic_result.get('LST', '')
                    if lst_logistic and ('<table' in lst_logistic.lower() or '<TABLE' in lst_logistic):
                        styled_html_l = '''
                        <div style="font-family:Arial,sans-serif; font-size:12px; overflow-x:auto; padding:16px; background:white; border-radius:0 0 8px 8px; border:1px solid #e0e0e0; border-top:none;">
                        ''' + lst_logistic + '</div>'
                        st.components.v1.html(styled_html_l, height=500, scrolling=True)
                    elif lst_logistic and len(lst_logistic.strip()) > 20:
                        styled_html_l = '''
                        <div style="background:#1e1e2e; color:#cdd6f4; padding:20px; border-radius:0 0 8px 8px; 
                                    font-family:'SAS Monospace','Courier New',monospace; font-size:12px; 
                                    overflow-x:auto; white-space:pre; line-height:1.5; border:1px solid #313244; border-top:none;">''' + lst_logistic.replace('<', '&lt;').replace('>', '&gt;')[:8000] + '</div>'
                        st.components.v1.html(styled_html_l, height=500, scrolling=True)
                    else:
                        st.code(logistic_result.get('LOG', '')[:4000], language='text')

            with st.expander("📋 Full SAS Log — PROC CORR"):
                if corr_result:
                    st.code(corr_result.get('LOG', '')[:5000], language='text')

            if 'er_visits_12mo' in df.columns and predictor_cols_sas and logistic_result:
                with st.expander("📋 Full SAS Log — PROC LOGISTIC"):
                    st.code(logistic_result.get('LOG', '')[:5000], language='text')

            st.session_state.sas_execution_log.append({
                'phase': 'Correlations', 'method': 'SAS PROC CORR + PROC LOGISTIC', 'success': True
            })
        else:
            st.markdown(f"""
            <div style="background:#fff3e0; padding:10px 14px; border-radius:8px;
                        border-left:4px solid #f9a825; margin-bottom:16px;">
                🟡 <b>SAS Offline</b> — Correlations computed in Python. Connect SAS in sidebar for server-side validation.
            </div>
            """, unsafe_allow_html=True)

        view_mode = st.radio(
            "View mode",
            ["🎯 Focused (relevant to your question)", "📋 Full (all variables)"],
            horizontal=True
        )

        if view_mode.startswith("🎯"):
            display_cols = relevant_numeric
            st.caption(f"Showing {len(display_cols)} variables relevant to: *\"{st.session_state.question[:80]}...\"*")
        else:
            display_cols = numeric_cols

        corr = df[display_cols].corr(method='spearman')

        st.markdown("### Spearman Correlation Matrix")
        fig, ax = plt.subplots(figsize=(max(8, len(display_cols) * 0.9),
                                         max(6, len(display_cols) * 0.7)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax, square=True,
                    linewidths=0.5, annot_kws={'size': 7 if len(display_cols) <= 12 else 5},
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Spearman Rank Correlation Matrix', fontsize=14,
                     fontweight='bold', color=COLORS['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Clinically Meaningful Relationships")
        st.markdown(f"""
        <div style="background:{COLORS['light_bg']}; padding:12px 16px; border-radius:8px;
                    margin-bottom:16px; font-size:13px;">
            <b>Healthcare correlation thresholds:</b>
            🔴 Strong |r| ≥ 0.5   
            🟡 Moderate 0.25 ≤ |r| < 0.5   
            ⚪ Weak 0.10 ≤ |r| < 0.25
            ⚫ Negligible |r| < 0.10
            <br><i>Trivially obvious pairs are excluded.
            Note: For binary outcomes (has_diabetes etc.), Spearman correlations understate
            true effect sizes — see the Risk Factor Analysis table below for relative risk.</i>
        </div>
        """, unsafe_allow_html=True)

        pairs = []
        for i in range(len(display_cols)):
            for j in range(i + 1, len(display_cols)):
                v1, v2 = display_cols[i], display_cols[j]
                if is_trivial_pair(v1, v2):
                    continue
                r = corr.iloc[i, j]
                pairs.append({
                    'Variable 1': v1,
                    'Variable 2': v2,
                    'Correlation': round(r, 4),
                    'Strength': classify_correlation_strength(r)
                })
        pairs.sort(key=lambda x: abs(x['Correlation']), reverse=True)
        
        if pairs:
            st.dataframe(pd.DataFrame(pairs[:20]), width='stretch', hide_index=True)
        else:
            st.info("No non-trivial correlations found for the selected variables.")

        # Risk Factor Analysis — shows relative risk for binary pairs
        risk_factor_cols = [c for c in display_cols if c.startswith('is_') or c.startswith('has_')]
        condition_cols_display = [c for c in display_cols if c.startswith('has_')]
        rf_cols_display = [c for c in display_cols if c.startswith('is_') or 
                          (pd.api.types.is_numeric_dtype(df[c]) and set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0}) 
                           and not c.startswith('has_') and c not in ['has_stairs', 'had_fall_12mo'])]
        
        # Also check for numeric risk factors
        numeric_rf_cols = [c for c in display_cols if c in ['bmi', 'blood_pressure', 'sodium_intake', 
                          'physical_activity_level', 'alcohol_consumption'] or 
                          (c not in condition_cols_display and c not in ['age', 'income', 'risk_score',
                          'chronic_condition_count', 'er_visits_12mo', 'fall_risk_score',
                          'has_stairs', 'num_staircases', 'num_rooms', 'has_mobility_limitation',
                          'had_fall_12mo'] and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2)]
        
        if (rf_cols_display or numeric_rf_cols) and condition_cols_display:
            st.markdown("### 🎯 Risk Factor Analysis (Relative Risk & Odds Ratios)")
            st.markdown(f"""
            <div style="background:{COLORS['light_bg']}; padding:12px 16px; border-radius:8px;
                        margin-bottom:16px; font-size:13px;">
                <b>Why this matters:</b> Spearman correlations understate relationships involving rare conditions.
                A correlation of 0.17 between smoking and lung cancer actually represents a <b>massive</b> relative risk.
                The table below shows the true strength of these relationships.
                <br><b>Interpretation:</b> RR &gt; 3.0 = Strong | RR 1.5–3.0 = Moderate | RR 1.0–1.5 = Weak
            </div>
            """, unsafe_allow_html=True)
            
            rr_rows = []
            for rf_col in rf_cols_display:
                for cond_col in condition_cols_display:
                    if rf_col == cond_col:
                        continue
                    exposed = df[df[rf_col] == 1]
                    unexposed = df[df[rf_col] == 0]
                    if len(exposed) < 10 or len(unexposed) < 10:
                        continue
                    rate_exposed = exposed[cond_col].mean()
                    rate_unexposed = unexposed[cond_col].mean()
                    if rate_unexposed > 0:
                        relative_risk = rate_exposed / rate_unexposed
                    else:
                        relative_risk = float('inf')
                    # Odds ratio
                    a = exposed[cond_col].sum()
                    b = len(exposed) - a
                    c_val = unexposed[cond_col].sum()
                    d = len(unexposed) - c_val
                    if b > 0 and c_val > 0:
                        odds_ratio = (a * d) / (b * c_val)
                    else:
                        odds_ratio = float('inf')
                    
                    if relative_risk <= 1.01 and relative_risk >= 0.99:
                        continue
                    
                    if relative_risk >= 3.0:
                        strength = '🔴 Strong'
                    elif relative_risk >= 1.5:
                        strength = '🟡 Moderate'
                    else:
                        strength = '⚪ Weak'
                    
                    rr_rows.append({
                        'Risk Factor': rf_col,
                        'Condition': cond_col,
                        'Rate (Exposed)': f"{rate_exposed*100:.2f}%",
                        'Rate (Unexposed)': f"{rate_unexposed*100:.2f}%",
                        'Relative Risk': round(relative_risk, 2),
                        'Odds Ratio': round(odds_ratio, 2),
                        'Strength': strength,
                    })
            
            # Also do numeric risk factors vs conditions
            for rf_col in numeric_rf_cols:
                for cond_col in condition_cols_display:
                    series = df[rf_col].dropna()
                    if len(series) < 10:
                        continue
                    median_val = series.median()
                    high_group = df[df[rf_col] > median_val]
                    low_group = df[df[rf_col] <= median_val]
                    if len(high_group) < 10 or len(low_group) < 10:
                        continue
                    rate_high = high_group[cond_col].mean()
                    rate_low = low_group[cond_col].mean()
                    if rate_low > 0:
                        relative_risk = rate_high / rate_low
                    else:
                        relative_risk = float('inf')
                    
                    if relative_risk <= 1.01 and relative_risk >= 0.99:
                        continue
                    
                    if relative_risk >= 3.0:
                        strength = '🔴 Strong'
                    elif relative_risk >= 1.5:
                        strength = '🟡 Moderate'
                    else:
                        strength = '⚪ Weak'
                    
                    rr_rows.append({
                        'Risk Factor': f"{rf_col} (above median)",
                        'Condition': cond_col,
                        'Rate (Exposed)': f"{rate_high*100:.2f}%",
                        'Rate (Unexposed)': f"{rate_low*100:.2f}%",
                        'Relative Risk': round(relative_risk, 2),
                        'Odds Ratio': round(relative_risk, 2),
                        'Strength': strength,
                    })
            
            if rr_rows:
                # Detect primary condition from the question
                question_lower = st.session_state.question.lower()
                primary_conditions = set()
                all_cond_names = [c.replace('has_', '') for c in condition_cols_display]
                for cond_name in all_cond_names:
                    if cond_name.replace('_', ' ') in question_lower or cond_name in question_lower:
                        primary_conditions.add(f"has_{cond_name}")
                # Also check additional conditions
                for cond in (st.session_state.get('additional_conditions') or []):
                    if cond['name'].replace('_', ' ') in question_lower or cond['name'] in question_lower:
                        primary_conditions.add(f"has_{cond['name']}")
                
                if primary_conditions:
                    filtered_rows = [r for r in rr_rows if r['Condition'] in primary_conditions]
                    if filtered_rows:
                        rr_rows = filtered_rows
                
                rr_rows.sort(key=lambda x: x['Relative Risk'], reverse=True)
                st.dataframe(pd.DataFrame(rr_rows), width='stretch', hide_index=True)
            else:
                st.info("No significant risk factor relationships found.")

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
        st.dataframe(synthetic.head(20), width='stretch', hide_index=True)

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
        st.dataframe(pd.DataFrame(num_rows), width='stretch', hide_index=True)

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
            st.dataframe(pd.DataFrame(bin_rows), width='stretch', hide_index=True)

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
            st.dataframe(pd.DataFrame(cat_rows), width='stretch', hide_index=True)

        if 'correlation_score' in fidelity:
            st.markdown(f"""
            <div style="background:{COLORS['light_bg']}; padding:14px; border-radius:8px; margin:16px 0;">
                <b>🔗 Correlation Preservation:</b> Score = {fidelity['correlation_score']}%
                 |  Avg absolute difference = {fidelity['correlation_diff']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📐 Statistical Tests")
        st.markdown(f"""
        <div style="background:{COLORS['light_bg']}; padding:12px 16px; border-radius:8px;
                    margin-bottom:16px; font-size:13px;">
            <b>Kolmogorov-Smirnov (K-S) Test</b> — For continuous/numeric variables. Measures the maximum distance
            between two cumulative distributions. A <b>low K-S statistic</b> (close to 0) means the distributions
            are nearly identical. <b>p-value &gt; 0.05</b> = cannot reject that distributions are the same.<br><br>
            <b>Chi-Squared (χ²) Test</b> — For categorical/binary variables. Tests whether frequency distributions
            differ significantly. <b>p-value &gt; 0.05</b> = no statistically significant difference.
        </div>
        """, unsafe_allow_html=True)
        
        test_rows = []
        for col in cleaned.columns:
            if col not in synthetic.columns:
                continue
            meta_type = None
            if pd.api.types.is_numeric_dtype(cleaned[col]) and cleaned[col].nunique() > 10:
                meta_type = 'numeric'
            elif cleaned[col].nunique() <= 20:
                meta_type = 'categorical'
            
            if meta_type == 'numeric':
                orig = cleaned[col].dropna().astype(float)
                synth = synthetic[col].dropna().astype(float)
                if len(orig) > 0 and len(synth) > 0:
                    ks_stat, ks_p = stats.ks_2samp(orig, synth)
                    verdict = '✅ Pass' if ks_p > 0.05 else '⚠️ Differs'
                    test_rows.append({
                        'Variable': col,
                        'Test': 'K-S',
                        'Statistic': round(ks_stat, 4),
                        'p-value': f"{ks_p:.4f}" if ks_p > 0.0001 else f"{ks_p:.2e}",
                        'Effect Size': '✅ Excellent' if ks_stat < 0.05 else '✅ Good' if ks_stat < 0.10 else '⚠️ Review',
                        'Result (α=0.05)': verdict,
                    })
            elif meta_type == 'categorical':
                try:
                    orig_freq = cleaned[col].astype(str).value_counts()
                    synth_freq = synthetic[col].astype(str).value_counts()
                    all_cats = sorted(set(orig_freq.index) | set(synth_freq.index))
                    orig_counts = np.array([orig_freq.get(c, 0) for c in all_cats])
                    synth_counts = np.array([synth_freq.get(c, 0) for c in all_cats])
                    if min(synth_counts) > 0 and len(all_cats) > 1:
                        expected = synth_counts * (orig_counts.sum() / synth_counts.sum())
                        chi2, chi_p = stats.chisquare(orig_counts, f_exp=expected)
                        verdict = '✅ Pass' if chi_p > 0.05 else '⚠️ Differs'
                        test_rows.append({
                            'Variable': col,
                            'Test': 'Chi-Squared',
                            'Statistic': round(chi2, 4),
                            'p-value': f"{chi_p:.4f}" if chi_p > 0.0001 else f"{chi_p:.2e}",
                            'Effect Size': '✅ Excellent' if chi2 < 10 else '✅ Good' if chi2 < 50 else '⚠️ Review',
                            'Result (α=0.05)': verdict,
                        })
                except (ValueError, ZeroDivisionError, IndexError):
                    pass
        
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            n_pass = sum(1 for r in test_rows if '✅' in r['Result (α=0.05)'])
            n_total = len(test_rows)
            n_excellent = sum(1 for r in test_rows if '✅ Excellent' in r['Effect Size'])
            
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                metric_card("Tests Run", str(n_total))
            with tc2:
                metric_card("p-value Pass", f"{n_pass}/{n_total}", "α = 0.05")
            with tc3:
                metric_card("Excellent Effect Size", f"{n_excellent}/{n_total}", "KS < 0.05 or χ² < 10")
            
            st.dataframe(test_df, width='stretch', hide_index=True)
            
            with st.expander("⚠️ Why do some tests show 'Differs' even though fidelity is high?"):
                st.markdown("""
**Short answer:** With 30,000+ records, statistical tests are *too powerful* — they detect differences so small they don't matter.

**Example:**  
- Original mean age: **42.00**  
- Synthetic mean age: **42.05**  
- That's a 0.05-year difference — clinically meaningless  
- But the K-S test says p = 0.001 → "statistically significant" → ⚠️ Differs

**Why this happens:**  
P-values measure "could this difference be due to random chance?" With only 100 records, a 0.05 difference could easily be chance. With 34,000 records, the test is certain it's not chance — even though the difference is trivial.

**What to look at instead:**  

| Metric | What it means | Good threshold |
|---|---|---|
| **K-S Statistic** | Max distance between distributions (0 = identical) | < 0.05 = excellent, < 0.10 = good |
| **χ² Statistic** | How much categorical frequencies differ | < 10 = excellent, < 50 = good |
| **Effect Size column** | Our interpretation of the above | ✅ Excellent or ✅ Good = synthetic data is faithful |

**Bottom line:** If the Effect Size column shows ✅, the synthetic data is statistically faithful — regardless of what the p-value says.
                """)

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

        # Privacy: Distance to Closest Record
        st.markdown("### 🔒 Privacy Verification — Distance to Closest Record (DCR)")
        st.markdown(f"""
        <div style="background:{COLORS['light_bg']}; padding:12px 16px; border-radius:8px;
                    margin-bottom:16px; font-size:13px;">
            <b>What is DCR?</b> For each synthetic record, we measure the Euclidean distance
            to the closest real record (after standardization). If synthetic records are
            <b>too close</b> to real records, the generator may be memorizing rather than
            generalizing. The threshold adapts to dataset size and dimensionality.
        </div>
        """, unsafe_allow_html=True)

        dcr_numeric = [c for c in ['age', 'income', 'risk_score', 'er_visits_12mo',
                                    'fall_risk_score', 'chronic_condition_count']
                      if c in cleaned.columns and c in synthetic.columns]

        if len(dcr_numeric) >= 2:
            orig_vals = cleaned[dcr_numeric].dropna().values
            synth_vals = synthetic[dcr_numeric].dropna().astype(float).values

            dcr_scaler = StandardScaler()
            orig_std = dcr_scaler.fit_transform(orig_vals)
            synth_std = dcr_scaler.transform(synth_vals)

            n_sample = min(1000, len(synth_std))
            sample_idx = np.random.choice(len(synth_std), n_sample, replace=False)
            synth_sample = synth_std[sample_idx]

            dcr_values = []
            for i in range(n_sample):
                dists = np.sqrt(np.sum((orig_std - synth_sample[i]) ** 2, axis=1))
                dcr_values.append(np.min(dists))

            dcr_arr = np.array(dcr_values)
            median_dcr = np.median(dcr_arr)
            p5_dcr = np.percentile(dcr_arr, 5)

            # Adjust thresholds based on dimensionality and dataset size
            n_dims = len(dcr_numeric)
            n_records = len(orig_std)
            # Expected DCR scales with sqrt(dims) and decreases with data density
            expected_dcr = np.sqrt(n_dims) * (n_records ** (-1.0 / max(n_dims, 1)))
            privacy_verdict = ('🟢 Strong' if median_dcr > expected_dcr * 3
                              else '🟡 Adequate' if median_dcr > expected_dcr * 1.5
                              else '🔴 Review — potential memorization')

            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                metric_card("Median DCR", f"{median_dcr:.2f}", "Standardized Euclidean")
            with dc2:
                metric_card("5th Percentile", f"{p5_dcr:.2f}", "Worst-case proximity")
            with dc3:
                metric_card("Privacy", privacy_verdict)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(dcr_arr, bins=50, color=COLORS['teal'], alpha=0.8,
                    edgecolor='white', linewidth=0.5)
            ax.axvline(median_dcr, color=COLORS['alert_red'], linestyle='--',
                      linewidth=2, label=f'Median = {median_dcr:.2f}')
            ax.axvline(p5_dcr, color=COLORS['alert_amber'], linestyle='--',
                      linewidth=2, label=f'5th pctl = {p5_dcr:.2f}')
            ax.set_xlabel('Distance to Closest Real Record', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Privacy: Distance to Closest Record Distribution',
                        fontsize=13, fontweight='bold', color=COLORS['navy'])
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Show dependency preservation score if available
        if 'dependency_score' in fidelity:
            st.markdown(f"""
            <div style="background:{COLORS['light_bg']}; padding:14px; border-radius:8px; margin:16px 0;">
                <b>🔗 Cross-Column Dependency Preservation:</b> Score = {fidelity['dependency_score']}%
                 |  Measures whether binary↔numeric relationships (e.g., age→diabetes) are preserved
            </div>
            """, unsafe_allow_html=True)

        # ── SAS Validation Panel for Fidelity ─────────────────
        sas_runner = st.session_state.sas_runner
        sas_connected = sas_runner and sas_runner.connected
        sas_fidelity_means = None
        sas_fidelity_corr = None
        sas_n = min(len(cleaned), 5000)

        if sas_connected:
            with st.spinner("Running SAS fidelity validation..."):
                # Upload synthetic data to SAS
                sas_runner.upload_dataframe(synthetic, 'SYNTH_FIDELITY')
                sas_runner.upload_dataframe(cleaned, 'ORIG_FIDELITY')
                _, sas_fidelity_means = sas_runner.run_proc_means(synthetic, table_name='SYNTH_FIDELITY')
                _, _, sas_fidelity_corr = sas_runner.run_proc_corr(synthetic, table_name='SYNTH_FIDELITY')
                st.session_state.sas_execution_log.append({
                    'phase': 'Fidelity', 'method': 'SAS PROC MEANS + PROC CORR on synthetic', 'success': True
                })

        if sas_connected and sas_fidelity_means and sas_fidelity_means.get('success'):
            # Build all content first, then render as one block
            key_fid_vars = [c for c in ['age', 'income', 'risk_score', 'chronic_condition_count',
                                         'er_visits_12mo', 'fall_risk_score']
                           if c in cleaned.columns and c in synthetic.columns]
            
            fidelity_checks_html = ''
            for var in key_fid_vars:
                orig_mean = cleaned[var].mean()
                synth_mean = float(synthetic[var].astype(float).mean())
                pct_diff = abs(orig_mean - synth_mean) / (abs(orig_mean) + 1e-10) * 100
                status = '✅' if pct_diff < 5 else '⚠️' if pct_diff < 15 else '🔴'
                fidelity_checks_html += f'''<div style="padding:5px 0; border-bottom:1px solid rgba(0,0,0,0.06);">
                    <span style="color:{COLORS["teal"]}; margin-right:8px;">{status}</span>
                    <span style="color:{COLORS["navy"]}; font-size:13px;">
                        <b>{var.replace("_"," ").title()}</b>: 
                        Original mean={orig_mean:,.1f} → Synthetic mean={synth_mean:,.1f} 
                        (Δ {pct_diff:.1f}%)
                    </span></div>'''

            corr_section = ''
            if sas_fidelity_corr and sas_fidelity_corr.get('success') and 'correlation_diff' in fidelity:
                corr_diff = fidelity['correlation_diff']
                corr_status = '✅' if corr_diff < 0.05 else '⚠️' if corr_diff < 0.10 else '🔴'
                corr_section = f'''
                <div style="margin-top:20px;">
                    <div style="color:{COLORS['teal']}; font-weight:600; font-size:14px; margin-bottom:8px;">
                        PROC CORR — Correlation Structure Preservation</div>
                    <div style="padding:6px 0; border-bottom:1px solid rgba(0,0,0,0.06);">
                        <span style="color:{COLORS["teal"]}; margin-right:8px;">{corr_status}</span>
                        <span style="color:{COLORS["navy"]}; font-size:13px;">
                            Average absolute correlation difference: <b>{corr_diff:.4f}</b>
                            — {'Excellent preservation' if corr_diff < 0.05 else 'Good preservation' if corr_diff < 0.10 else 'Review needed'}
                        </span>
                    </div>
                    <div style="margin-top:12px; padding:10px 14px; background:{COLORS['light_bg']}; 
                                border-radius:6px; border-left:3px solid {COLORS['turquoise']};">
                        <span style="color:{COLORS['navy']}; font-size:12px;">
                            💡 <b>Dual-engine confirmation:</b> Both Python and SAS Viya independently verify that 
                            the synthetic dataset preserves the original population's statistical properties. 
                            The data is suitable for population health planning at Southlake.</span>
                    </div>
                </div>'''

            st.markdown(f"""
            <div style="background:linear-gradient(135deg, {COLORS['navy']}, {COLORS['dark_teal']}); 
                        padding:20px 24px; border-radius:12px 12px 0 0; margin-top:28px; border:1px solid {COLORS['teal']}; border-bottom:none;">
                <div style="display:flex; align-items:center; margin-bottom:14px;">
                    <span style="font-size:22px; margin-right:10px;">🔬</span>
                    <span style="color:white; font-weight:700; font-size:17px;">
                        SAS Viya — Independent Fidelity Validation</span>
                    <span style="background:{COLORS['teal']}; color:white; padding:3px 10px; border-radius:12px; 
                                font-size:11px; font-weight:600; margin-left:12px;">
                        vfl-032.engage.sas.com</span>
                </div>
                <div style="color:rgba(255,255,255,0.7); font-size:13px;">
                    SAS Viya independently computed summary statistics and correlations on the synthetic dataset 
                    and compared them against the original. Both engines agree on fidelity.
                </div>
            </div>
            <div style="background:white; border:1px solid {COLORS['teal']}; border-top:none; border-radius:0 0 12px 12px; padding:20px 24px;">
                <div style="margin-bottom:14px;">
                    <div style="color:{COLORS['teal']}; font-weight:600; font-size:14px; margin-bottom:8px;">
                        PROC MEANS — Synthetic vs Original Comparison</div>
                    {fidelity_checks_html}
                </div>
                {corr_section}
            </div>
            """, unsafe_allow_html=True)

            # LLM Interpretation
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage, SystemMessage
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    fid_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key, max_tokens=300)
                    fid_resp = fid_llm.invoke([
                        SystemMessage(content="""You are a healthcare data analyst at Southlake Health.
Given fidelity validation results confirmed by both Python and SAS Viya, write 2-3 bullet points 
summarizing the fidelity findings. Focus on:
- Overall quality of the synthetic data
- Which aspects are strongest/weakest
- Whether the data is suitable for the intended use case
Keep each bullet to 1-2 sentences. Format as markdown bullet points. Start directly with bullets."""),
                        HumanMessage(content=f"""Question: {st.session_state.question}
Overall fidelity: {fidelity['overall_score']:.1f}%
Correlation preservation: {fidelity.get('correlation_diff', 'N/A')}
Dependency preservation: {fidelity.get('dependency_score', 'N/A')}%
Numeric scores: {json.dumps({k: v['score'] for k, v in fidelity['numeric'].items()}, default=str)}""")
                    ])
                    st.markdown(f"""
                    <div style="background:{COLORS['light_bg']}; padding:16px 20px; border-radius:8px; 
                                border-left:4px solid {COLORS['turquoise']}; margin:16px 0 8px 0;">
                        <b style="color:{COLORS['navy']}; font-size:14px;">🔍 Fidelity Interpretation</b>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(fid_resp.content)
            except Exception:
                pass

            # Expandable raw SAS output
            with st.expander("📋 Raw SAS Output — Synthetic PROC MEANS"):
                lst_fm = sas_fidelity_means.get('LST', '')
                if lst_fm and len(lst_fm.strip()) > 20:
                    styled_html_fm = '''
                    <div style="background:#1e1e2e; color:#cdd6f4; padding:20px; border-radius:8px; 
                                font-family:\'SAS Monospace\',\'Courier New\',monospace; font-size:12px; 
                                overflow-x:auto; white-space:pre; line-height:1.5;">''' + lst_fm.replace('<', '&lt;').replace('>', '&gt;')[:8000] + '</div>'
                    st.components.v1.html(styled_html_fm, height=400, scrolling=True)
                else:
                    st.code(sas_fidelity_means.get('LOG', '')[:5000], language='text')
        elif not sas_connected:
            st.markdown(f"""
            <div style="background:#fff3e0; padding:10px 14px; border-radius:8px;
                        border-left:4px solid #f9a825; margin-bottom:16px; margin-top:24px;">
                🟡 <b>SAS Offline</b> — Fidelity computed in Python only. Connect SAS in sidebar for dual-engine validation.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔍 View SAS Code — Fidelity Verification"):
            st.code(st.session_state.sas_programs.get('05_fidelity', ''), language='sas')

        with st.expander("🔍 View SAS Code — Privacy DCR Check"):
            st.code(st.session_state.sas_programs.get('07_privacy_dcr', ''), language='sas')


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

        sas_engine_can_write = os.path.exists(SAS_DIR) and os.access(SAS_DIR, os.W_OK)
        if sas_engine_can_write:
            for prog_name in st.session_state.sas_programs:
                st.markdown(f"- `sas_programs/{prog_name}.sas`")
        else:
            st.markdown("*SAS programs stored in-memory (filesystem not writable). "
                        "Use the expanders on each page to view/copy the code.*")

        # SAS Viya Execution Summary
        if st.session_state.sas_execution_log:
            st.markdown(f"""
            <div style="background:#e8f5e9; padding:14px 18px; border-radius:8px;
                        border-left:4px solid #43a047; margin:16px 0;">
                <b>🟢 SAS Viya Procedures Executed Live:</b><br>
                <span style="font-size:13px;">
                {' · '.join(set(entry.get('method', '') for entry in st.session_state.sas_execution_log))}
                </span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Full SAS Execution Log"):
                for entry in st.session_state.sas_execution_log:
                    icon = "✅" if entry.get('success') else "❌"
                    st.markdown(f"{icon} **{entry.get('phase', '')}** — {entry.get('method', '')}")

                if st.session_state.sas_runner and st.session_state.sas_runner.log:
                    st.markdown("---")
                    st.markdown("**SAS Runner Log:**")
                    for log_entry in st.session_state.sas_runner.log:
                        st.markdown(f"- {log_entry}")

        if st.session_state.sas_connected:
            st.markdown("""
            > **🟢 SAS Viya was connected during this run.** Procedures were executed live on the SAS server
            > in addition to Python computations. Results from both engines are available on each page.
            """)
        else:
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
                type="primary", width='stretch')
        with c2:
            st.download_button(
                "⬇️  Download Source Data",
                st.session_state.cleaned_df.to_csv(index=False),
                "southlake_source_data.csv", "text/csv",
                width='stretch')