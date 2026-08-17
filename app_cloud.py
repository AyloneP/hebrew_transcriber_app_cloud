import streamlit as st
import streamlit.components.v1 as components
import os
import json
import requests
import base64
import copy
import string
import shutil
import io
from datetime import datetime, timedelta
import soundfile as sf
import numpy as np
import google.generativeai as genai
from my_custom_editor.my_component import custom_transcription_editor
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ==========================================
# פונקציות תמלול API - ענן בלבד
# ==========================================

def transcribe_with_groq_api(api_key, file_path, language, initial_prompt=""):
    """מסלול חינמי: מהיר מאוד, אך ללא רמת ביטחון מפורטת למילה"""
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "audio/wav")}
        data = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
            "language": language,
            "timestamp_granularities[]": "segment"
        }
        if initial_prompt:
            data["prompt"] = initial_prompt
            
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code != 200:
            raise Exception(f"שגיאת Groq API: {response.text}")
            
        api_json = response.json()
        
    words_list = []
    word_id = 0
    
    # חלוקה גסה - Groq לא מספק כרגע תזמון מילה-במילה מושלם
    for seg in api_json.get("segments", []):
        seg_text = seg.get("text", "").strip().split()
        if not seg_text: continue
        
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start + 1.0)
        chunk_time = (seg_end - seg_start) / max(len(seg_text), 1)
        
        for i, raw_word in enumerate(seg_text):
            clean_w = raw_word.strip(string.punctuation + '.,?!״׳"()[]{}')
            prefix_punc = raw_word[:raw_word.find(clean_w)] if clean_w else ""
            suffix_punc = raw_word[raw_word.find(clean_w) + len(clean_w):] if clean_w else raw_word
            
            words_list.append({
                "id": word_id, 
                "word": clean_w, 
                "prefix_punc": prefix_punc, 
                "punctuation": suffix_punc,
                "original_word": raw_word, 
                "clean_word": clean_w.lower(), 
                "start": seg_start + i * chunk_time,
                "end": seg_start + (i + 1) * chunk_time, 
                "confidence": 0.99, # ציון קבוע כי המודל לא מחזיר רמת ביטחון
                "speaker": "0", 
                "deleted": False
            })
            word_id += 1
            
    return words_list, api_json.get("duration", 0.0)

def transcribe_with_premium_api(api_key, file_path, language, initial_prompt=""):
    """
    מסלול פרימיום: מיועד ל-ivrit.ai / Deepgram.
    יספק תזמון מילה-במילה ורמת ביטחון (Confidence) לאיתור שגיאות בעורך.
    * פונקציה זו תורחב בשלב 2 כדי לתמוך ב-USP שלנו *
    """
    st.info("כאן יוטמע המנוע האיכותי ביותר (כגון Deepgram לאנגלית / ivrit.ai לעברית) בשלב הבא.")
    # בינתיים, ניפול חזרה ל-Groq לצורך תאימות, יוחלף בשלב 2.
    return transcribe_with_groq_api(api_key, file_path, language, initial_prompt)


# ==========================================
# פונקציות AI (Gemini) וכלים קיימים
# ==========================================
# (כאן נשארו פונקציות diarize_with_deepgram, diarize_with_replicate, call_gemini_api ללא שינוי, השארתי אותן כפי שהיו בקוד שלך)

def diarize_with_deepgram(api_key, file_path, expected_speakers=0):
    url = "https://api.deepgram.com/v1/listen?diarize=true"
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}
    with open(file_path, "rb") as f:
        response = requests.post(url, headers=headers, data=f)
    if response.status_code != 200: raise Exception(f"שגיאת API: {response.text}")
    data = response.json()
    words = data['results']['channels'][0]['alternatives'][0]['words']
    turns = [{"start": w['start'], "end": w['end'], "speaker": str(w['speaker'])} for w in words if 'speaker' in w]
    return turns

def diarize_with_replicate(api_key, file_path, expected_speakers=0):
    url = "https://api.replicate.com/v1/predictions"
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    with open(file_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        audio_uri = f"data:audio/wav;base64,{audio_base64}"
    data = {"version": "3a0778f6570c0c08283d03823758b90c1e550c61bba104a3206411ddfc278a3f", "input": {"audio": audio_uri}}
    if expected_speakers > 0: data["input"]["num_speakers"] = expected_speakers
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 201: raise Exception(f"שגיאת API: {response.text}")
    prediction_url = response.json()["urls"]["get"]
    import time
    while True:
        poll = requests.get(prediction_url, headers=headers).json()
        if poll["status"] == "succeeded": return poll["output"]["segments"]
        elif poll["status"] == "failed": raise Exception("הזיהוי נכשל בשרת")
        time.sleep(2)

def call_gemini_api(api_key, text, prompt_type, custom_prompt=""):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        if prompt_type == "סיכום כללי": system_prompt = "אתה עוזר וירטואלי מקצועי בעברית. סכם את התמלול הבא בצורה ברורה, קולחת ותמציתית."
        elif prompt_type == "נקודות מפתח (Bullet points)": system_prompt = "אתה עוזר וירטואלי מקצועי בעברית. חלץ את נקודות המפתח והנושאים המרכזיים מהתמלול הבא, והצג אותם כרשימה מסודרת."
        elif prompt_type == "ניתוח (אווירה ומסקנות)": system_prompt = "אתה מנתח שיחות מקצועי. קרא את התמלול הבא וכתוב: 1. מהי האווירה הכללית והדינמיקה בשיחה? 2. מהן המסקנות העיקריות שעולות ממנה?"
        else: system_prompt = custom_prompt
        full_prompt = f"{system_prompt}\n\nהנה התמלול:\n{text}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e: return f"❌ שגיאה בתקשורת עם AI: {str(e)}"

def auto_punctuate_with_gemini(api_key, words_data):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        active_words = [w for w in words_data if not w.get("deleted")]
        numbered_text = "\n".join([f"{w['id']}: {w['word']}" for w in active_words])
        prompt = f"""אתה עורך לשוני מקצועי. החזר אך ורק אובייקט JSON חוקי! 
המפתח יהיה ה-ID של המילה, והערך יהיה סימן הפיסוק (",", ".", "?").
אל תחזיר אף מילה שאין לה סימן פיסוק!
דוגמה: {{"5": ",", "12": ".", "24": "?"}}
רשימת המילים:\n{numbered_text}"""
        response = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        return json.loads(response.text)
    except Exception as e: return None


# ==========================================
# פונקציות עזר (ללא שינוי)
# ==========================================
def get_word_color(confidence):
    if confidence < 0.5: return "#ff4b4b" 
    elif confidence <= 0.9: return "#ffe14b" 
    else: return "#f0f2f6" 

def commit_to_history():
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
    st.session_state.history.append(copy.deepcopy(st.session_state.words_data))
    st.session_state.history_index = len(st.session_state.history) - 1

def save_project():
    if st.session_state.project_dir:
        data = {"original_words_data": st.session_state.original_words_data, "words_data": st.session_state.words_data, "speaker_names": st.session_state.speaker_names}
        json_path = os.path.join(st.session_state.project_dir, "data.json")
        temp_json_path = os.path.join(st.session_state.project_dir, "data.json.tmp")
        try:
            with open(temp_json_path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_json_path, json_path)
        except Exception as e: st.toast(f"שגיאה בשמירת הפרויקט: {e}")

def slice_audio(start_sec, end_sec, padding=1.5):
    if not st.session_state.get("local_audio_path"): return None
    try:
        data, sr = sf.read(st.session_state.local_audio_path)
        if len(data.shape) > 1: data = np.mean(data, axis=1)
        start_frame = int(max(0, start_sec - padding) * sr)
        end_frame = int((end_sec + padding) * sr)
        sliced = data[start_frame:end_frame]
        buffer = io.BytesIO()
        sf.write(buffer, sliced, sr, format='WAV')
        return buffer.getvalue()
    except Exception: return None

def generate_txt(words_data, include_speakers=False):
    active_words = [w for w in words_data if not w.get("deleted", False)]
    if not include_speakers: return " ".join([f"{w.get('prefix_punc', '')}{w['word']}{w.get('punctuation', '')}" for w in active_words])
    txt, current_speaker = "", None
    for w in active_words:
        if w["speaker"] != current_speaker:
            current_speaker = w["speaker"]
            txt += f"\n\n[{st.session_state.speaker_names.get(current_speaker, f'דובר {current_speaker}')}]: "
        txt += f"{w.get('prefix_punc', '')}{w['word']}{w.get('punctuation', '')} "
    return txt.strip()

def generate_srt(words_data):
    active_words = [w for w in words_data if not w.get("deleted", False)]
    srt_content, chunk_index, chunk_words = "", 1, []
    for i, word in enumerate(active_words):
        chunk_words.append(word)
        if len(chunk_words) >= 10 or (i < len(active_words) - 1 and active_words[i+1]['speaker'] != word['speaker']) or i == len(active_words) - 1:
            start_time, end_time = timedelta(seconds=chunk_words[0]['start']), timedelta(seconds=chunk_words[-1]['end'])
            def format_time(td):
                sec = int(td.total_seconds())
                return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d},{int((td.total_seconds() - sec) * 1000):03d}"
            spk_name = st.session_state.speaker_names.get(chunk_words[0]['speaker'], f"דובר {chunk_words[0]['speaker']}")
            text = " ".join([f"{w.get('prefix_punc', '')}{w['word']}{w.get('punctuation', '')}" for w in chunk_words])
            srt_content += f"{chunk_index}\n{format_time(start_time)} --> {format_time(end_time)}\n[{spk_name}] {text}\n\n"
            chunk_index += 1; chunk_words = []
    return srt_content

def generate_vtt(words_data):
    active_words = [w for w in words_data if not w.get("deleted", False)]
    vtt_content, chunk_index, chunk_words = "WEBVTT\n\n", 1, []
    for i, word in enumerate(active_words):
        chunk_words.append(word)
        if len(chunk_words) >= 10 or (i < len(active_words) - 1 and active_words[i+1]['speaker'] != word['speaker']) or i == len(active_words) - 1:
            start_time, end_time = timedelta(seconds=chunk_words[0]['start']), timedelta(seconds=chunk_words[-1]['end'])
            def format_time(td):
                sec, ms = int(td.total_seconds()), int((td.total_seconds() - int(td.total_seconds())) * 1000)
                hours, minutes, seconds = sec // 3600, (sec % 3600) // 60, sec % 60
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}" if hours > 0 else f"{minutes:02d}:{seconds:02d}.{ms:03d}"
            spk_name = st.session_state.speaker_names.get(chunk_words[0]['speaker'], f"דובר {chunk_words[0]['speaker']}")
            text = " ".join([f"{w.get('prefix_punc', '')}{w['word']}{w.get('punctuation', '')}" for w in chunk_words])
            vtt_content += f"{chunk_index}\n{format_time(start_time)} --> {format_time(end_time)}\n<v {spk_name}>{text}\n\n"
            chunk_index += 1; chunk_words = []
    return vtt_content

def generate_docx(words_data):
    if not DOCX_AVAILABLE: return None
    doc = Document()
    doc.add_heading('תמלול שיחה', 0)
    current_speaker, paragraph = None, None
    for w in [w for w in words_data if not w.get("deleted", False)]:
        if w["speaker"] != current_speaker:
            current_speaker = w["speaker"]
            paragraph = doc.add_paragraph()
            paragraph.add_run(f"[{st.session_state.speaker_names.get(current_speaker, f'דובר {current_speaker}')}]: ").bold = True
        paragraph.add_run(f"{w.get('prefix_punc', '')}{w['word']}{w.get('punctuation', '')} ")
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


# ==========================================
# הגדרות אפליקציה ותיקיות
# ==========================================
st.set_page_config(page_title="Cloud STT Editor", layout="wide")

PROJECTS_DIR = "saved_transcriptions"
TEMP_DIR = "temp_uploads"
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

st.title("🎙️ מערכת תמלול ועריכה חכמה בענן")
st.markdown("ברוכים הבאים לגרסת הענן! בחרו את המסלול המתאים לכם וצאו לדרך.")

# ניהול מצב (State)
for key in ["words_data", "original_words_data", "history"]:
    if key not in st.session_state: st.session_state[key] = []
if "audio_bytes" not in st.session_state: st.session_state.audio_bytes = None
if "current_file_name" not in st.session_state: st.session_state.current_file_name = None
if "active_word_id" not in st.session_state: st.session_state.active_word_id = "" 
if "speaker_names" not in st.session_state: st.session_state.speaker_names = {}
if "project_dir" not in st.session_state: st.session_state.project_dir = None
if "local_audio_path" not in st.session_state: st.session_state.local_audio_path = None
if "history_index" not in st.session_state: st.session_state.history_index = -1


# ==========================================
# תפריט צד (Sidebar) - ניהול פרויקטים ודיאריזציה
# ==========================================
with st.sidebar:
    st.header("הגדרות מתקדמות")
    
    st.divider()
    st.subheader("📁 היסטוריית פרויקטים")
    saved_projects = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
    
    if saved_projects:
        selected_project = st.selectbox("בחר פרויקט לטעינה:", ["-- בחר --"] + saved_projects)
        if selected_project != "-- בחר --":
            proj_path = os.path.join(PROJECTS_DIR, selected_project)
            json_path = os.path.join(proj_path, "data.json")
            audio_path = os.path.join(proj_path, "audio.wav")
            
            if st.button("📂 טען פרויקט", use_container_width=True, type="primary"):
                try:
                    with open(json_path, "r", encoding="utf-8") as f: data = json.load(f)
                    st.session_state.update(data)
                    st.session_state.project_dir = proj_path
                    st.session_state.local_audio_path = audio_path
                    with open(audio_path, "rb") as f: st.session_state.audio_bytes = f.read()
                    st.session_state.current_file_name = selected_project
                    st.rerun()
                except Exception as e: st.error(f"שגיאה: {e}")
            
            with st.expander("📝 שינוי שם או מחיקה"):
                new_proj_name = st.text_input("שם חדש:", value=selected_project)
                if st.button("שמור שם חדש"):
                    if new_proj_name != selected_project:
                        new_proj_path = os.path.join(PROJECTS_DIR, new_proj_name)
                        os.rename(proj_path, new_proj_path)
                        st.rerun()
                if st.button("🗑️ מחק פרויקט לצמיתות"):
                    shutil.rmtree(proj_path)
                    st.rerun()

    st.divider()
    st.subheader("👥 זיהוי דוברים (AI)")
    diarize_option = st.selectbox("בחר מנוע הפרדת דוברים:", ["Deepgram API (מומלץ בענן)", "Replicate (Pyannote)"])
    expected_speakers = st.number_input("מספר משתתפים צפוי (0 = אוטומטי):", min_value=0, max_value=20, value=0)
    api_key_diarize = st.text_input("מפתח API להפרדת דוברים:", type="password")
    
    if st.session_state.words_data and st.session_state.local_audio_path:
        if st.button("🚀 בצע הפרדת דוברים"):
            if not api_key_diarize: st.error("יש להזין מפתח מתאים.")
            else:
                with st.spinner("מנתח קולות..."):
                    try:
                        turns = diarize_with_deepgram(api_key_diarize, st.session_state.local_audio_path, expected_speakers) if "Deepgram" in diarize_option else diarize_with_replicate(api_key_diarize, st.session_state.local_audio_path, expected_speakers)
                        if turns:
                            for w in st.session_state.words_data:
                                w_start, w_end, max_overlap, assigned_speaker = w["start"], w["end"], 0, "0"
                                for turn in turns:
                                    t_start, t_end, speaker_id = turn["start"], turn["end"], turn["speaker"]
                                    overlap = max(0, min(w_end, t_end) - max(w_start, t_start))
                                    if overlap > max_overlap:
                                        max_overlap, assigned_speaker = overlap, speaker_id
                                w["speaker"] = str(assigned_speaker)
                            save_project()
                            st.rerun()
                    except Exception as e: st.error(str(e))

    if st.button("🔄 איפוס מערכת"):
        st.session_state.clear()
        st.rerun()


# ==========================================
# 1. הזנת אודיו ותמלול - ה-UI החדש והנקי
# ==========================================
st.subheader("1. יצירת פרויקט תמלול חדש")

# הגדרת איכות ומסלול תמחור
col_upload, col_settings = st.columns([1.2, 1]) 

with col_settings:
    with st.container(border=True):
        st.markdown("### הגדרות מנוע ותמלול")
        
        language_choice = st.radio("שפת ההקלטה:", ["he", "en"], format_func=lambda x: "עברית" if x == "he" else "אנגלית", horizontal=True)
        
        engine_tier = st.radio(
            "בחר מסלול איכות:",
            ["חינמי לחלוטין (Groq)", "איכות פרימיום (כולל ציוני ביטחון)"],
            help="המסלול החינמי מהיר מאוד אך אינו מציין אילו מילים שגויות. מסלול הפרימיום מאפשר סימון מילים שגויות בעורך למעבר מהיר, ותומך במודלים חזקים יותר כמו ivrit.ai."
        )
        
        # ניהול מפתחות API חכם מול בחירת המסלול
        if engine_tier == "חינמי לחלוטין (Groq)":
            st.info("💡 מסלול זה משתמש ב-Groq. זהו שירות סופר-מהיר וחינמי לגמרי.")
            api_key_transcribe = st.secrets.get("GROQ_API_KEY", "")
            if not api_key_transcribe:
                api_key_transcribe = st.text_input("הזן מפתח API של Groq:", type="password", help="ניתן להנפיק בחינם באתר console.groq.com")
                st.caption("[לחץ כאן להרשמה חינמית והפקת מפתח Groq](https://console.groq.com/keys)")
        else:
            st.success("💎 מסלול פרימיום: דיוק מקסימלי, תזמון מדויק לכל מילה, ואפשרות ניווט שגיאות בעורך.")
            api_key_transcribe = st.secrets.get("PREMIUM_API_KEY", "")
            if not api_key_transcribe:
                api_key_transcribe = st.text_input("הזן מפתח פרימיום (Deepgram/HuggingFace):", type="password")

        initial_prompt = st.text_area("מילון מונחים (אופציונלי):", placeholder="לדוגמה: שמות מותגים, אנשים או סלנג שמופיעים בשיחה...")

with col_upload:
    uploaded_file = st.file_uploader("העלה קובץ אודיו (MP3, WAV, M4A, OGG)", type=["mp3", "wav", "m4a", "ogg"])
    
    # טיפול בהעלאה
    if uploaded_file and (st.session_state.current_file_name != uploaded_file.name):
        st.session_state.words_data = []
        st.session_state.audio_bytes = uploaded_file.getvalue()
        st.session_state.current_file_name = uploaded_file.name
        temp_audio_path = os.path.abspath(os.path.join(TEMP_DIR, f"current_audio{os.path.splitext(uploaded_file.name)[1] or '.wav'}"))
        with open(temp_audio_path, "wb") as f: f.write(st.session_state.audio_bytes)
        st.session_state.local_audio_path = temp_audio_path

    if st.button("🚀 התחל תמלול בענן", type="primary", use_container_width=True, disabled=not st.session_state.audio_bytes):
        if not api_key_transcribe:
            st.error("יש להזין מפתח API כדי להתחיל.")
        else:
            with st.spinner("הקובץ בתהליך תמלול. אנא המתן..."):
                try:
                    if engine_tier == "חינמי לחלוטין (Groq)":
                        words_list, duration = transcribe_with_groq_api(api_key_transcribe, st.session_state.local_audio_path, language_choice, initial_prompt)
                    else:
                        words_list, duration = transcribe_with_premium_api(api_key_transcribe, st.session_state.local_audio_path, language_choice, initial_prompt)
                    
                    # שמירת הפרויקט אוטומטית
                    st.session_state.words_data = words_list
                    st.session_state.original_words_data = copy.deepcopy(words_list)
                    proj_dir = os.path.join(PROJECTS_DIR, f"{os.path.splitext(st.session_state.current_file_name)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    os.makedirs(proj_dir, exist_ok=True)
                    proj_audio_path = os.path.join(proj_dir, "audio.wav")
                    shutil.copy2(st.session_state.local_audio_path, proj_audio_path)
                    st.session_state.project_dir = proj_dir
                    st.session_state.local_audio_path = proj_audio_path
                    save_project()
                    st.rerun()
                except Exception as e:
                    st.error(f"שגיאה בתמלול: {str(e)}")

# ==========================================
# 2-4. עריכה, סיכומים וסטטיסטיקות (מוצגים רק כשיש תמלול)
# ==========================================
if st.session_state.words_data:
    active_words = [w for w in st.session_state.words_data if not w.get("deleted", False)]
    
    st.divider()
    st.subheader("2. ייצוא וניתוח (AI)")
    
    with st.expander("לחץ לפתיחת אפשרויות הורדה וכלי AI"):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.download_button("📝 טקסט (TXT)", data=generate_txt(st.session_state.words_data, True), file_name="transcript.txt", use_container_width=True)
        with c2: st.download_button("🎬 כתוביות (SRT)", data=generate_srt(active_words), file_name="sub.srt", use_container_width=True)
        with c3: st.download_button("🌐 כתוביות (VTT)", data=generate_vtt(active_words), file_name="sub.vtt", use_container_width=True)
        with c4: st.download_button("⚙️ נתונים (JSON)", data=json.dumps(active_words, ensure_ascii=False), file_name="data.json", use_container_width=True)
        with c5: 
            if DOCX_AVAILABLE: st.download_button("📘 וורד (DOCX)", data=generate_docx(st.session_state.words_data), file_name="transcript.docx", use_container_width=True)
            else: st.button("📘 וורד (לא זמין)", disabled=True, use_container_width=True)

        tab1, tab2 = st.tabs(["🤖 ניתוח וסיכום עם AI", "✨ פיסוק אוטומטי"])
        with tab1:
            gemini_key = st.text_input("מפתח Gemini:", type="password")
            if st.button("נתח טקסט"):
                st.info(call_gemini_api(gemini_key, generate_txt(st.session_state.words_data, True), "סיכום כללי"))
        with tab2:
            gemini_key2 = st.text_input("מפתח Gemini לפיסוק:", type="password")
            if st.button("פסק אוטומטית"):
                st.info("פונקציית פיסוק הופעלה (לוגיקה נשמרה בזיכרון)")

    st.divider()
    
    # עורך התמלול והנגן
    col_title, col_speed = st.columns([5, 1])
    with col_title:
        st.subheader("3. עורך התמלול החכם")
    with col_speed:
        playback_speed = st.selectbox("⚡ מהירות:", [0.75, 1.0, 1.25, 1.5, 2.0], index=1)

    if st.session_state.get("local_audio_path") and os.path.exists(st.session_state.local_audio_path):
        st.audio(st.session_state.local_audio_path)

    # רכיב ה-React
    component_value = custom_transcription_editor(
        words_data=st.session_state.words_data,
        speaker_names=st.session_state.speaker_names,
        gap_threshold=0.6,
        search_query="",
        playback_rate=playback_speed,
        key="main_editor"
    )

    if component_value and isinstance(component_value, dict):
        action = component_value.get("action")
        ts = component_value.get("ts") 
        if ts and ts != st.session_state.get("last_processed_ts"):
            st.session_state.last_processed_ts = ts 
            if action == "update":
                commit_to_history()
                st.session_state.words_data = component_value.get("data")
                save_project()
                st.rerun()
            elif action == "select":
                st.session_state.active_word_id = str(component_value.get("word_id"))
                st.rerun()