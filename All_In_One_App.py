import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import spacy
from spacy import displacy
import pandas as pd
from faker import Faker
import networkx as nx
import matplotlib.pyplot as plt
from translate import Translator
from summarizer import Summarizer
import itertools
import base64
import io
import json
from datetime import datetime
import textwrap

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment

# Load models and data
@dash.callback(Output('dummy', 'children'), Input('dummy', 'children'))
def load_models():
    global nlp, nlp_deidentify, summarizer_model, fake
    global translator_to_english, translator_to_norwegian
    global icd10_data, atc_df, ddi_data, renal_dataset
    
    nlp = spacy.load('Model/NER_Model')
    nlp_deidentify = spacy.load('nb_core_news_sm')
    summarizer_model = Summarizer('distilbert-base-uncased')
    fake = Faker()
    
    translator_to_english = Translator(from_lang="no", to_lang="en")
    translator_to_norwegian = Translator(from_lang="en", to_lang="no")
    
    icd10_data = pd.read_csv('Datasets/ICD_Names.csv')
    icd10_data['diagnosis'] = icd10_data['diagnosis'].str.lower()
    atc_df = pd.read_csv('Datasets/ATC_Injector.csv')
    ddi_data = pd.read_csv('Datasets/DDIs_2_Columns.csv')
    renal_dataset = pd.read_csv('Datasets/Drug_Dose_In_Renal_Impairment_To_Use.csv')

# Helper functions (same as your original code)
def image_to_base64(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return ""

def recognize_entities(text, entity_types):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in entity_types:
            entities.append((ent.text, ent.label_))
    return entities, doc

def extract_diagnoses(text):
    doc = nlp(text)
    diagnoses = [ent.text for ent in doc.ents if ent.label_ == 'CONDITION']
    return diagnoses

def get_icd_codes_and_diagnoses(diagnosis, icd10_data):
    diagnosis = diagnosis.lower()
    matching_rows = icd10_data[icd10_data['diagnosis'].str.contains(diagnosis)]
    if not matching_rows.empty:
        icd_codes_and_diagnoses = list(matching_rows.itertuples(index=False, name=None))
        return icd_codes_and_diagnoses
    else:
        return None

def assign_icd_codes(text, icd10_data):
    diagnoses = extract_diagnoses(text)
    icd_codes_and_diagnoses = {diagnosis: get_icd_codes_and_diagnoses(diagnosis, icd10_data) for diagnosis in diagnoses}
    return icd_codes_and_diagnoses

def correct_icd_codes(text, icd10_data):
    icd_codes_and_diagnoses = assign_icd_codes(text, icd10_data)
    return icd_codes_and_diagnoses

def summarize_text(text, summary_proportion=0.5):
    if text:
        try:
            translated_text_parts = []
            for piece in textwrap.wrap(text, 500):
                translated_piece = translator_to_english.translate(piece)
                translated_text_parts.append(translated_piece)
            translated_text = " ".join(translated_text_parts)
        except Exception as e:
            return f"Translation to English failed: {e}"

        try:
            summary_english = summarizer_model(translated_text, min_length=60, max_length=500)
            if isinstance(summary_english, list):
                summary_english = " ".join(summary_english)
        except Exception as e:
            return f"Summarization failed: {e}"

        try:
            summary_norwegian_parts = []
            for piece in textwrap.wrap(summary_english, 500):
                translated_piece = translator_to_norwegian.translate(piece)
                summary_norwegian_parts.append(translated_piece)
            summary_norwegian = " ".join(summary_norwegian_parts)
        except Exception as e:
            return f"Translation to Norwegian failed: {e}"

        return f'Summary: {summary_norwegian}'

def dose_check(text, renal_dataset):
    doc = nlp(text)
    drugs_normal = [ent.text for ent in doc.ents if ent.label_ == 'SUBSTANCE']
    drug_dose_map_normal = {}
    for drug in drugs_normal:
        matching_rows = renal_dataset[renal_dataset['Short_Drug'].str.lower() == drug.lower()]
        if not matching_rows.empty:
            dose_text = matching_rows["Normal dose"].iloc[0]
            dose_parts = dose_text.split(". ")
            formatted_dose = "\n".join([f"  - {part.strip()}." for part in dose_parts if part.strip()])
            drug_dose_map_normal[drug] = formatted_dose
    return drug_dose_map_normal

def get_renal_doses(text, renal_dataset):
    doc = nlp(text)
    drugs_renal = [ent.text for ent in doc.ents if ent.label_ == 'SUBSTANCE']
    drug_dose_map = {}
    for drug in drugs_renal:
        matching_rows = renal_dataset[renal_dataset['Short_Drug'].str.lower() == drug.lower()]
        if not matching_rows.empty:
            dose_text = matching_rows["Dose in renal impairment GFR (mL/min)"].iloc[0]
            dose_parts = dose_text.split(". ")
            formatted_dose = "\n".join([f"  - {part.strip()}." for part in dose_parts if part.strip()])
            drug_dose_map[drug] = formatted_dose
    return drug_dose_map

def atc_code_injection(text, atc_df):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['SUBSTANCE', 'CONDITION']]
    entity_atc_dict = {}
    for entity in entities:
        entity_lower = entity.lower()
        atc_code = get_atc_code(entity_lower, atc_df)
        if atc_code is not None and entity_lower not in entity_atc_dict:
            entity_atc_dict[entity_lower] = atc_code
    for entity, atc_code in entity_atc_dict.items():
        styled_atc_code = f'<mark style="background-color: green; color: white;">(ATC code: {atc_code})</mark>'
        text = text.replace(entity, f'{entity} {styled_atc_code}')
    return text

def get_atc_code(entity, atc_df):
    matching_rows = atc_df[(atc_df['virkestoff'] == entity) | (atc_df['varenavn'] == entity)]
    if not matching_rows.empty:
        return matching_rows['atckode'].iloc[0]
    else:
        return None

def patient_identification(text):
    doc = nlp_deidentify(text)
    identified_names = [ent.text for ent in doc.ents if ent.label_ == 'PER']
    identified_addresses = [ent.text for ent in doc.ents if ent.label_ == 'LOC']
    name_map = {name: fake.name() for name in identified_names}
    address_map = {address: fake.address() for address in identified_addresses}
    anonymized_data = text
    for real_name, fake_name in name_map.items():
        anonymized_data = anonymized_data.replace(real_name, f'<mark style="background-color: red; color: white;">{fake_name}</mark>')
    for real_address, fake_address in address_map.items():
        anonymized_data = anonymized_data.replace(real_address, f'<mark style="background-color: blue; color: white;">{fake_address}</mark>')
    return anonymized_data

def check_ddi(text, ddi_data):
    doc = nlp(text)
    substances = [ent.text.lower() for ent in doc.ents if ent.label_ == 'SUBSTANCE']
    if len(substances) < 2:
        return pd.DataFrame(), "Not enough substances found to check for interactions."
    results = []
    interactions = set()
    for substance1, substance2 in itertools.combinations(substances, 2):
        interaction = ddi_data[((ddi_data['lm1'] == substance1) & 
                                (ddi_data['lm2'] == substance2)) |
                               ((ddi_data['lm1'] == substance2) & 
                                (ddi_data['lm2'] == substance1))]
        for row in interaction.itertuples():
            interaction_key = (substance1, substance2, row.grad)
            if interaction_key not in interactions:
                interactions.add(interaction_key)
                results.append({"Substance 1": substance1, 
                                "Substance 2": substance2, 
                                "Grade": row.grad})
    
    ddi_df = pd.DataFrame(results)
    if ddi_df.empty:
        return pd.DataFrame(), "No drug-drug interactions found."
    return ddi_df, None

def visualize_comorbidity(text):
    doc = nlp(text)
    conditions_with_positions = [(ent.text, ent.start_char) for ent in doc.ents if ent.label_ == 'CONDITION']
    conditions = []
    seen = set()
    for condition, _ in sorted(conditions_with_positions, key=lambda x: x[1]):
        if condition not in seen:
            seen.add(condition)
            conditions.append(condition)
    if not conditions:
        return None
    graph = nx.DiGraph()
    for i in range(len(conditions)-1):
        graph.add_edge(conditions[i], conditions[i+1])
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9, with_labels=True)
    plt.axis('off')
    plt.tight_layout()
    
    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

# Layout
image_base64 = image_to_base64("cartoon.JPG")

app.layout = dbc.Container([
    # Hidden div for loading models
    html.Div(id='dummy', style={'display': 'none'}),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(src=f"data:image/jpeg;base64,{image_base64}", 
                        style={'border-radius': '50%', 'width': '40px', 'height': '40px'}),
                html.Div("Developed by: Mohsen Askar", 
                        style={'font-size': '10px', 'text-align': 'center'})
            ], style={'position': 'absolute', 'top': '10px', 'right': '10px', 
                     'text-align': 'center'})
        ])
    ]),
    
    # Title
    dbc.Row([
        dbc.Col([
            html.H1("AMER tool 📋 (Experimental)", className="text-center mt-4"),
            html.P("Automatic Medical Entities Recognizer (AMER) tool offers several functionalities to process Norwegian medical text using NLP techniques.",
                  className="text-center text-muted", style={'font-size': '12px'})
        ])
    ]),
    
    # Sidebar (using Offcanvas)
    dbc.Row([
        dbc.Col([
            dbc.Button("Modules Description", id="open-offcanvas", n_clicks=0, className="mb-3"),
            dbc.Offcanvas([
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.P("Recognizes medical entities in the text such as drugs (substance name), diagnoses, etc. You can specify which type of entities to show or to show them all.")
                    ], title="1. 💊 Recognize Entities"),
                    dbc.AccordionItem([
                        html.P("Suggests the correct ICD codes for the diagnoses mentioned in the text.")
                    ], title="2. 📋 Correct ICD Codes"),
                    dbc.AccordionItem([
                        html.P("Inserts the correct ATC codes to the substances and drug names in the text.")
                    ], title="3. 💉 Inject ATC Codes"),
                    dbc.AccordionItem([
                        html.P("Identifies patient's sensitive info. Inserts fake names, dates, etc. in the patient records.")
                    ], title="4. 🕵️ Deidentify Patient's Information"),
                    dbc.AccordionItem([
                        html.P("Checks for possible drug-drug interactions between the drugs mentioned in the text. Red: severe, Orange: moderate, Green: possible DDIs.")
                    ], title="5. ⚠️ Check DDIs"),
                    dbc.AccordionItem([
                        html.P("Draws a directed network of the comorbidities progression in chronological order from the text.")
                    ], title="6. 📈 Visualize Comorbidity Progression"),
                    dbc.AccordionItem([
                        html.P("Returns the correct drug dosage for the identified drugs in the text. Source of doses: Renal Drug Handbook, fifth edition.")
                    ], title="7. 🧪 Dose Checker"),
                    dbc.AccordionItem([
                        html.P("Returns the correct drug dosage in case of renal impairment. Source of doses: Renal Drug Handbook, fifth edition.")
                    ], title="8. 💊⚠️ Dose Checker in Renal Impairment"),
                    dbc.AccordionItem([
                        html.P("Summarizes the patient records to 50% of the original text.")
                    ], title="9. 📝 EHR Summarizer"),
                    dbc.AccordionItem([
                        html.P("Divides the EHR into episodes of hospital admissions. Extracts the most relevant information from each episode.")
                    ], title="10. 🏥 Structure EHR (soon)"),
                    dbc.AccordionItem([
                        html.P("Identifies potential side effects in the text.")
                    ], title="11. 🚑 Find Side Effects (soon)"),
                ], start_collapsed=True)
            ], id="offcanvas", is_open=False, title="Modules Description"),
        ], width=12)
    ]),
    
    # Input area
    dbc.Row([
        dbc.Col([
            dcc.Textarea(
                id='ehr-input',
                placeholder='Paste the EHR text here',
                style={'width': '100%', 'height': 200},
                className="mb-3"
            )
        ], width=12)
    ]),
    
    # Entity checkboxes
    dbc.Row([
        dbc.Col([
            html.Label("Select the entity type:"),
            dbc.Row([
                dbc.Col([dbc.Checkbox(id='check-all', label='All')], width=3),
                dbc.Col([dbc.Checkbox(id='check-condition', label='CONDITION')], width=3),
                dbc.Col([dbc.Checkbox(id='check-substance', label='SUBSTANCE')], width=3),
                dbc.Col([dbc.Checkbox(id='check-physiology', label='PHYSIOLOGY')], width=3),
            ]),
            dbc.Row([
                dbc.Col([dbc.Checkbox(id='check-procedure', label='PROCEDURE')], width=4),
                dbc.Col([dbc.Checkbox(id='check-anat', label='ANAT_LOC')], width=4),
                dbc.Col([dbc.Checkbox(id='check-micro', label='MICROORGANISM')], width=4),
            ], className="mt-2")
        ], width=12, className="mb-3")
    ]),
    
    # Action buttons
    dbc.Row([
        dbc.Col([dbc.Button('Recognize Entities', id='btn-recognize', color='primary', className='w-100')], width=4),
        dbc.Col([dbc.Button('Correct ICD Codes', id='btn-icd', color='secondary', className='w-100')], width=4),
        dbc.Col([dbc.Button('Inject ATC Codes', id='btn-atc', color='info', className='w-100')], width=4),
    ], className="mb-2"),
    
    dbc.Row([
        dbc.Col([dbc.Button('Deidentifier', id='btn-deidentify', color='warning', className='w-100')], width=4),
        dbc.Col([dbc.Button('Check DDIs', id='btn-ddi', color='danger', className='w-100')], width=4),
        dbc.Col([dbc.Button('Visualize Comorbidity', id='btn-comorbidity', color='success', className='w-100')], width=4),
    ], className="mb-2"),
    
    dbc.Row([
        dbc.Col([dbc.Button('Dose Checker', id='btn-dose', color='primary', className='w-100')], width=4),
        dbc.Col([dbc.Button('Correct Dose Renal', id='btn-renal', color='secondary', className='w-100')], width=4),
        dbc.Col([dbc.Button('Summarize text', id='btn-summarize', color='info', className='w-100')], width=4),
    ], className="mb-2"),
    
    dbc.Row([
        dbc.Col([dbc.Button('Structure text (soon)', id='btn-structure', color='warning', className='w-100', disabled=True)], width=6),
        dbc.Col([dbc.Button('Find Side Effects (soon)', id='btn-side-effects', color='danger', className='w-100', disabled=True)], width=6),
    ], className="mb-4"),
    
    # Output area
    dbc.Row([
        dbc.Col([
            html.Div(id='output-area', className='mt-4')
        ], width=12)
    ]),
    
    # Footer with visitor count
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Div(id='visitor-count', className='text-center text-muted', 
                    style={'padding': '10px', 'font-size': '14px'}),
            html.Div(datetime.now().strftime("%B %d, %Y"), 
                    className='text-center text-muted', 
                    style={'font-size': '12px'})
        ])
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    State("offcanvas", "is_open"),
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    [Output('check-condition', 'value'),
     Output('check-substance', 'value'),
     Output('check-physiology', 'value'),
     Output('check-procedure', 'value'),
     Output('check-anat', 'value'),
     Output('check-micro', 'value')],
    Input('check-all', 'value'),
    prevent_initial_call=True
)
def toggle_all_checkboxes(all_checked):
    if all_checked:
        return [True] * 6
    return [False] * 6

@app.callback(
    Output('output-area', 'children'),
    [Input('btn-recognize', 'n_clicks'),
     Input('btn-icd', 'n_clicks'),
     Input('btn-atc', 'n_clicks'),
     Input('btn-deidentify', 'n_clicks'),
     Input('btn-ddi', 'n_clicks'),
     Input('btn-comorbidity', 'n_clicks'),
     Input('btn-dose', 'n_clicks'),
     Input('btn-renal', 'n_clicks'),
     Input('btn-summarize', 'n_clicks')],
    [State('ehr-input', 'value'),
     State('check-all', 'value'),
     State('check-condition', 'value'),
     State('check-substance', 'value'),
     State('check-physiology', 'value'),
     State('check-procedure', 'value'),
     State('check-anat', 'value'),
     State('check-micro', 'value')],
    prevent_initial_call=True
)
def process_action(btn_rec, btn_icd, btn_atc, btn_deid, btn_ddi, btn_comorb, 
                   btn_dose, btn_renal, btn_summ, text, all_check, cond_check, 
                   subst_check, phys_check, proc_check, anat_check, micro_check):
    
    if not text:
        return html.Div("Please enter some text first.", className="alert alert-warning")
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Determine entity types
    entity_types = []
    if all_check:
        entity_types = ["ANAT_LOC", "CONDITION", "MICROORGANISM", "PHYSIOLOGY", "PROCEDURE", "SUBSTANCE"]
    else:
        if anat_check: entity_types.append("ANAT_LOC")
        if cond_check: entity_types.append("CONDITION")
        if micro_check: entity_types.append("MICROORGANISM")
        if phys_check: entity_types.append("PHYSIOLOGY")
        if proc_check: entity_types.append("PROCEDURE")
        if subst_check: entity_types.append("SUBSTANCE")
    
    try:
        if button_id == 'btn-recognize':
            entities, doc = recognize_entities(text, entity_types)
            if len(entities) > 0:
                entity_df = pd.DataFrame([(e[0], e[1]) for e in entities], columns=['Entity', 'Type'])
                colors = {"ANAT_LOC": "#FAC748", "CONDITION": "#FF5733", "MICROORGANISM": "#47D1D1", 
                         "PHYSIOLOGY": "#2E86C1", "PROCEDURE": "#BB8FCE", "SUBSTANCE": "#27AE60"}
                options = {"ents": entity_types, "colors": colors}
                html_viz = displacy.render(doc, style="ent", options=options)
                
                return html.Div([
                    html.H5(f"Number of recognized entities: {len(entities)}"),
                    html.H6("Recognized entities:"),
                    dash_table.DataTable(
                        data=entity_df.to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in entity_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'}
                    ),
                    html.Div(html.Iframe(srcDoc=html_viz, style={'width': '100%', 'height': '400px', 'border': 'none'}))
                ])
            else:
                return html.Div("No entities found.", className="alert alert-info")
        
        elif button_id == 'btn-icd':
            result = correct_icd_codes(text, icd10_data)
            return html.Div([
                html.H5("ICD Codes:"),
                html.Pre(json.dumps(result, indent=2))
            ])
        
        elif button_id == 'btn-atc':
            result = atc_code_injection(text, atc_df)
            return html.Div(dcc.Markdown(result, dangerously_allow_html=True))
        
        elif button_id == 'btn-deidentify':
            result = patient_identification(text)
            return html.Div(dcc.Markdown(result, dangerously_allow_html=True))
        
        elif button_id == 'btn-ddi':
            ddi_df, msg = check_ddi(text, ddi_data)
            if msg:
                return html.Div(msg, className="alert alert-info")
            color_dict = {1: 'red', 2: 'orange', 3: 'green'}
            ddi_df['Color'] = ddi_df['Grade'].map(color_dict)
            return html.Div([
                html.H5("Drug-Drug Interactions:"),
                dash_table.DataTable(
                    data=ddi_df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in ['Substance 1', 'Substance 2', 'Grade']],
                    style_data_conditional=[
                        {'if': {'filter_query': '{Grade} = 1'}, 'color': 'red', 'fontWeight': 'bold'},
                        {'if': {'filter_query': '{Grade} = 2'}, 'color': 'orange', 'fontWeight': 'bold'},
                        {'if': {'filter_query': '{Grade} = 3'}, 'color': 'green', 'fontWeight': 'bold'},
                    ]
                )
            ])
        
        elif button_id == 'btn-comorbidity':
            img_base64 = visualize_comorbidity(text)
            if img_base64:
                return html.Div([
                    html.H5("Comorbidity Progression:"),
                    html.Img(src=f'data:image/png;base64,{img_base64}', style={'width': '100%'})
                ])
            else:
                return html.Div("No conditions found to visualize.", className="alert alert-info")
        
        elif button_id == 'btn-dose':
            result = dose_check(text, renal_dataset)
            if result:
                output = [html.H5("Correct Doses:")]
                for drug, dose in result.items():
                    output.append(html.H6(f"{drug}:"))
                    output.append(html.Pre(dose))
                return html.Div(output)
            else:
                return html.Div("No drugs detected in the text.", className="alert alert-info")
        
        elif button_id == 'btn-renal':
            result = get_renal_doses(text, renal_dataset)
            if result:
                output = [html.H5("Correct Doses for Renal Impairment according to GFR value:")]
                for drug, dose in result.items():
                    output.append(html.H6(f"{drug}:"))
                    output.append(html.Pre(dose))
                return html.Div(output)
            else:
                return html.Div("No drugs detected in the text.", className="alert alert-info")
        
        elif button_id == 'btn-summarize':
            result = summarize_text(text)
            return html.Div([
                html.H5("Summary:"),
                html.P(result)
            ])
        
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")
    
    return html.Div()

@app.callback(
    Output('visitor-count', 'children'),
    Input('dummy', 'children')
)
def update_visitor_count(_):
    # Simple local file-based counter (for development)
    try:
        try:
            with open('visitor_count.txt', 'r') as f:
                count = int(f.read().strip())
        except FileNotFoundError:
            count = 0
        
        count += 1
        
        with open('visitor_count.txt', 'w') as f:
            f.write(str(count))
        
        return f"👥 Total Visitors: {count}"
    except:
        return "👥 Total Visitors: N/A"

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
