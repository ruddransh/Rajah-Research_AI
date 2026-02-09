import os
import datetime
import time
import io
import json
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- LANGCHAIN & GEMINI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# --- GOOGLE DRIVE IMPORTS ---
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ==========================================
# 🛑 CONFIGURATION
# ==========================================
# For Local Testing: Paste Key below.
# For Render Deployment: Set this in "Environment Variables" settings.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "PASTE_YOUR_GEMINI_KEY_HERE")

CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

app = Flask(__name__)
CORS(app)  # Allows Chrome Extension to talk to this server

# ==========================================
# 1. STATELESS AUTHENTICATION (NEW)
# ==========================================
def get_drive_service_from_token(token):
    """
    Builds the Drive service using the token sent by the Chrome Extension.
    This allows ANY user to save files to THEIR OWN Drive.
    """
    try:
        creds = Credentials(token=token)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return None

def save_to_drive(service, df, filename_prefix="Research_Updates"):
    """Auto-saves the dataframe as a CSV to Google Drive."""
    try:
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"{filename_prefix}_{timestamp}.csv"

        file_metadata = {'name': filename, 'mimeType': 'text/csv'}
        media = MediaIoBaseUpload(csv_buffer, mimetype='text/csv', resumable=True)
        
        file = service.files().create(
            body=file_metadata, media_body=media, fields='id, webViewLink'
        ).execute()
        
        return file.get('webViewLink')
    except Exception as e:
        print(f"❌ Failed to save to Drive: {e}")
        return None

# ==========================================
# 2. CLINICAL TRIALS SCOUT
# ==========================================
def fetch_recent_trials(disease_query, months_back=6):
    today = datetime.date.today()
    start_date_threshold = today - datetime.timedelta(days=30*months_back)
    
    statuses = ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"]
    status_string = ",".join(statuses)

    params = {
        "query.term": disease_query,       
        "filter.overallStatus": status_string,
        "pageSize": 300,                   
        "sort": "StudyFirstPostDate:desc"  
    }
    
    try:
        headers = {"User-Agent": "ResearchAgent/7.0"}
        response = requests.get(CLINICAL_TRIALS_API, params=params, headers=headers)
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return []
        data = response.json()
    except Exception as e:
        print(f"❌ Network Error: {e}")
        return []
    
    studies = []
    if 'studies' in data:
        for study in data['studies']:
            protocol = study.get('protocolSection', {})
            id_module = protocol.get('identificationModule', {})
            status_module = protocol.get('statusModule', {})
            design_module = protocol.get('designModule', {})
            cond_module = protocol.get('conditionsModule', {})
            int_module = protocol.get('armsInterventionsModule', {})
            loc_module = protocol.get('contactsLocationsModule', {})
            desc_module = protocol.get('descriptionModule', {})

            start_date_str = status_module.get('startDateStruct', {}).get('date')
            if not start_date_str: continue

            try:
                if len(start_date_str) == 7: dt = datetime.datetime.strptime(start_date_str, "%Y-%m").date()
                else: dt = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
                if dt < start_date_threshold: continue
            except: continue

            all_conditions = cond_module.get('conditions', [])
            matches = [c for c in all_conditions if disease_query.lower() in c.lower()]
            if not matches: continue 

            studies.append({
                "NCT Number": id_module.get('nctId'),
                "Study title": id_module.get('briefTitle'),
                "Study status": status_module.get('overallStatus'),
                "Condition": ", ".join(matches),  
                "Intervention": ", ".join([f"{i.get('name')}" for i in int_module.get('interventions', [])]) or "N/A",
                "Phase of study": ", ".join(design_module.get('phases', [])) if design_module.get('phases') else "Not Specified",
                "Enrollment number": design_module.get('enrollmentInfo', {}).get('count', 'N/A'),
                "Study type": design_module.get('studyType'),
                "Start date": start_date_str,
                "Completion Date": status_module.get('completionDateStruct', {}).get('date', 'N/A'),
                "Locations": ", ".join(list(set([l.get('country', '') for l in loc_module.get('locations', []) if l.get('country')])))[:50],
                "BriefSummary": desc_module.get('briefSummary', '')
            })

    return studies

# ==========================================
# 3. THE ANALYST (GEMINI)
# ==========================================
def analyze_trials_against_research(research_text, trials_list):
    if not trials_list: return pd.DataFrame()

    if "PASTE" in GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not set!")
        return pd.DataFrame()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )

    # Summarize the Research Context
    print("🧠 Summarizing research context...")
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the core hypothesis and goals of this research paper in 3 sentences: {text}"
    )
    chain = summary_prompt | llm
    research_summary = chain.invoke({"text": research_text[:25000]}).content 
    print(f"✅ Context: {research_summary}")

    analyzed_data = []
    BATCH_SIZE = 20
    
    analysis_prompt = ChatPromptTemplate.from_template("""
    MY RESEARCH SUMMARY: {research_summary}
    
    Here is a list of new clinical trials. 
    For EACH trial, determine if it is relevant to my research.
    
    TRIALS TO ANALYZE:
    {trials_text}
    
    OUTPUT REQUIREMENTS:
    Return a list strictly in this format, one line per trial:
    Trial_ID|Relevance(Yes/No)|Reason|Update Potential
    
    (Example: NCT12345|Yes|Matches JAK inhibitor target|New safety data)
    """)
    
    print(f"🔍 Analyzing {len(trials_list)} trials...")
    
    for i in range(0, len(trials_list), BATCH_SIZE):
        batch = trials_list[i : i + BATCH_SIZE]
        batch_text = ""
        for trial in batch:
            batch_text += f"ID: {trial['NCT Number']}\nTitle: {trial['Study title']}\nInterventions: {trial['Intervention']}\nSummary: {trial.get('BriefSummary', '')[:500]}\n---\n"
        
        try:
            chain = analysis_prompt | llm
            result = chain.invoke({
                "research_summary": research_summary, 
                "trials_text": batch_text
            })
            lines = result.content.strip().split('\n')
            results_map = {}
            for line in lines:
                parts = line.split('|')
                if len(parts) >= 4:
                    nct_id = parts[0].strip()
                    results_map[nct_id] = {'rel': parts[1].strip(), 'reason': parts[2].strip(), 'pot': parts[3].strip()}
            
            for trial in batch:
                res = results_map.get(trial['NCT Number'])
                if res:
                    trial['AI Relevance'] = res['rel']
                    trial['AI Reason'] = res['reason']
                    trial['AI Potential'] = res['pot']
                else:
                    trial['AI Relevance'] = "No" 
                    trial['AI Reason'] = "Analysis Skipped"
                    trial['AI Potential'] = "N/A"
                if 'BriefSummary' in trial: del trial['BriefSummary']
                analyzed_data.append(trial)
                
        except Exception as e:
            print(f"⚠️ Batch failed: {e}")
            for trial in batch:
                trial['AI Relevance'] = "Error"
                analyzed_data.append(trial)
        
        time.sleep(1) 

    return pd.DataFrame(analyzed_data)

# ==========================================
# 4. MAIN FLASK ROUTE
# ==========================================
@app.route('/analyze_active_tab', methods=['POST'])
def analyze_active_tab():
    # 1. Grab the token from the Header (Sent by Extension)
    user_token = request.headers.get('Authorization')
    
    # If using local testing without extension auth, you might need to handle this differently,
    # but for the shared version, this is required.
    if not user_token:
        # Fallback for local testing if you really need it, or just return error
        print("⚠️ No Token received from Extension.")
        # return jsonify({"status": "error", "message": "Login required."})

    data = request.json
    
    # 2. Get Data from Request
    page_text = data.get('page_text', '')
    disease = data.get('disease', 'Rheumatoid Arthritis')
    months = data.get('months', 6)  # Default to 6 if not sent
    
    print(f"📥 Received request: Disease='{disease}', Months={months}, Content Length={len(page_text)} chars")

    if len(page_text) < 50:
        return jsonify({"status": "error", "message": "Page content too short. Try a different tab."})

    # 3. Scout Clinical Trials
    trials = fetch_recent_trials(disease, months_back=months)
    
    if not trials:
        return jsonify({"status": "error", "message": f"No trials found in the last {months} months."})

    # 4. Analyze with Gemini
    df = analyze_trials_against_research(page_text, trials)
    
    if df.empty:
         return jsonify({"status": "error", "message": "AI Analysis returned no results."})

    # 5. SORTING (Not Filtering)
    # We create a helper column to sort "Yes" to the top, then "No"
    # 0 = Yes, 1 = No (Ascending sort puts 0 first)
    df['Sort_Rank'] = df['AI Relevance'].apply(lambda x: 0 if 'Yes' in str(x) else 1)
    final_df = df.sort_values(by=['Sort_Rank'])
    final_df = final_df.drop(columns=['Sort_Rank']) # Clean up

    # 6. Save to Drive using THEIR token
    drive_link = None
    
    if user_token:
        service = get_drive_service_from_token(user_token)
        if service:
            drive_link = save_to_drive(service, final_df, filename_prefix=f"Extension_{disease}")
        else:
            print("⚠️ Could not build Drive service from token.")
    else:
        print("⚠️ No token provided, skipping upload.")

    # Calculate match count for UI display
    match_count = len(final_df[final_df['AI Relevance'].str.contains("Yes", case=False, na=False)])

    return jsonify({
        "status": "success", 
        "message": "Analysis Complete!", 
        "drive_link": drive_link if drive_link else "#",
        "match_count": match_count
    })

if __name__ == '__main__':
    # Use PORT environment variable for Render, default to 5000 for local
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Research Agent Server Running on Port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)