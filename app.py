import io
import re
import tempfile
from pathlib import Path
from typing import List, Optional
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib
import sqlite3
from datetime import datetime
import json
import google.generativeai as genai


# path penting model dan db
VECTORIZER_PATH = Path("model/tfidf_vectorizer.joblib")
CLASSIFIER_PATH = Path("model/best_classifier.joblib")
LABEL_MAP = {0: "pengeluaran", 1: "pemasukan"}
DB_PATH = "finance_history.db"

st.set_page_config(page_title="FiceAI — Finance Tracker", layout="centered")


# fungsi database
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT NOT NULL,
            raw_text TEXT,
            label TEXT,
            amount REAL,
            description TEXT
        )
        """)
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def add_transaction_to_db(src: str, text: str, label: str, amount: float, description: str):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO transactions (source, raw_text, label, amount, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (src, text, label, amount, description),
        )
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Gagal menyimpan ke database: {e}")
    finally:
        if conn:
            conn.close()

def fetch_history() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        df = pd.read_sql_query("SELECT id, timestamp, source, label, amount, raw_text, description FROM transactions ORDER BY timestamp DESC", conn)
        return df
    except sqlite3.Error as e:
        st.error(f"Gagal mengambil data dari database: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def delete_transactions_by_ids(ids: list):
    if not ids:
        return
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        placeholders = ', '.join('?' for id in ids)
        query = f"DELETE FROM transactions WHERE id IN ({placeholders})"
        cursor.execute(query, ids)
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Gagal menghapus data: {e}")
    finally:
        if conn:
            conn.close()

def delete_all_transactions():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions")
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Gagal membersihkan database: {e}")
    finally:
        if conn:
            conn.close()

def calculate_current_balance() -> float:
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        df = pd.read_sql_query("SELECT label, amount FROM transactions", conn)
        conn.close()
        
        if df.empty:
            return 0.0
        pemasukan = df[df['label'] == 'pemasukan']['amount'].sum()
        pengeluaran = df[df['label'] == 'pengeluaran']['amount'].sum()
        return pemasukan - pengeluaran
    except Exception as e:
        st.error(f"Gagal menghitung saldo: {e}")
        return 0.0

@st.cache_data
def to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='History')
    processed_data = output.getvalue()
    return processed_data


# Util & Model

def analyze_receipt_with_gemini(api_key: str, image_bytes: bytes) -> Optional[dict]:
    """
    Menganalisis gambar struk menggunakan Gemini dan mengembalikan data terstruktur (JSON).
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        img = Image.open(io.BytesIO(image_bytes))

        # prompt gemini untuk mengekstrak teks dari gambar dan tampilan hasilnya hanya mengisi informasi yang dibuthkan
        prompt = """
        Analisis gambar ini dan berikan output dalam format JSON yang valid.
        Ekstrak informasi berikut:
        1. `nama_merchant`: Nama toko, restoran, bank, atau tempat transaksi terjadi. Ambil nama yang paling jelas.
        2. `total_belanja`: Angka total akhir yang harus dibayar (Grand Total).
        3. `teks_lengkap`: Semua teks yang bisa kamu baca dari gambar, sebagai satu string.

        Contoh output:
        {"nama_merchant": "PADHI RESTO", "total_belanja": 280253, "teks_lengkap": "PADHI RESTO Jl. Kol..."}

        Jika ada informasi yang tidak ditemukan, gunakan nilai null.
        """
        
        response = model.generate_content([prompt, img])
        
        # membersihkan output dari markdown backticks
        cleaned_response = re.sub(r'```json\s*|\s*```', '', response.text.strip())
        
        # Parsing JSON
        data = json.loads(cleaned_response)
        return data

    except Exception as e:
        st.error(f"Error saat menghubungi Gemini: {e}")
        return None

# fungsi-fungsi membersihkan text, memuat model, prediksi dll
def clean_text(s: str) -> str:
    return str(s).strip()

AMOUNT_PATTERNS = [
    r"(?<!\d)(\d{1,3}(?:[\.,]\d{3})+(?:[\.,]\d{2})?|\d+(?:[\.,]\d{2})?)(?!\d)",
    r"(\d+(?:[\.,]\d+)?)\s*(k|rb|ribu|jt|juta)\b",
]
UNIT_MULTIPLIER = {"k": 1_000, "rb": 1_000, "ribu": 1_000, "jt": 1_000_000, "juta": 1_000_000}

def _to_float(num_str: str) -> float:
    s = str(num_str).strip()
    s = re.sub(r'[^\d,.]', '', s)
    if s.endswith(',00') or s.endswith('.00'):
        s = s[:-3]
    s = s.replace('.', '').replace(',', '')
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan

def extract_amounts(text: str) -> List[float]:
    if not text:
        return []
    found = []
    for pat in AMOUNT_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            if m.lastindex and m.lastindex >= 2 and m.group(2):
                base = _to_float(m.group(1))
                unit = m.group(2).lower()
                mult = UNIT_MULTIPLIER.get(unit, 1)
                val = base * mult if base == base else np.nan
            else:
                val = _to_float(m.group(1) if m.lastindex else m.group(0))
            if val == val:
                found.append(val)
    uniq = []
    for v in found:
        if not any(abs(v - u) < 0.5 for u in uniq):
            uniq.append(v)
    return uniq

def pick_amount(amounts: List[float]) -> Optional[float]:
    return float(max(amounts)) if amounts else None

@st.cache_resource(show_spinner=True)
def load_models(vectorizer_path: Path, classifier_path: Path):
    if not Path(vectorizer_path).exists() or not Path(classifier_path).exists():
        st.warning(f"Model belum lengkap:\n- {vectorizer_path}\n- {classifier_path}")
        return None, None
    try:
        vectorizer = joblib.load(vectorizer_path)
        classifier = joblib.load(classifier_path)
        return vectorizer, classifier
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

vectorizer, classifier = load_models(VECTORIZER_PATH, CLASSIFIER_PATH)

def predict_sentence(raw_text: str) -> dict:
    if not vectorizer or not classifier:
        return {"ok": False, "error": "Model belum dimuat."}
    s_clean = clean_text(raw_text)
    try:
        X = vectorizer.transform([s_clean])
        pred = classifier.predict(X)[0]
        label = LABEL_MAP.get(int(pred), str(pred))
        amounts = extract_amounts(raw_text)
        amount = pick_amount(amounts)
        desc = re.sub(r"\s+", " ", re.sub(r"\b[\d\.,]+\s*(k|rb|ribu|jt|juta)?\b", "", raw_text, flags=re.IGNORECASE)).strip()
        return {
            "ok": True, "label": label, "amount": amount, "description": desc, "raw_text": raw_text
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    
def stt_transcribe_audio_bytes(audio_bytes: bytes, filename: str) -> str:
    """Mencoba mentranskripsi audio menggunakan Whisper, fallback ke Google Speech Recognition."""
    try:
        import whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        model = whisper.load_model("base") 
        result = model.transcribe(tmp_path, language="id")
        os.remove(tmp_path) # Hapus file sementara
        return (result.get("text") or "").strip()
    except Exception as e:
        st.info(f"Whisper tidak berhasil, mencoba metode lain... Error: {e}")
        pass 

    # Fallback menggunakan Google Speech Recognition
    try:
        import speech_recognition as sr
        from pydub import AudioSegment

        # Konversi audio ke format WAV yang kompatibel
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as raw:
            raw.write(audio_bytes)
            raw_path = raw.name
        
        sound = AudioSegment.from_file(raw_path)
        
        wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sound.export(wav_tmp.name, format="wav")

        r = sr.Recognizer()
        with sr.AudioFile(wav_tmp.name) as source:
            audio = r.record(source)
        
        text = r.recognize_google(audio, language="id-ID")
        
        # Hapus semua file sementara
        os.remove(raw_path)
        os.remove(wav_tmp.name)

        return text.strip()
    except Exception:
        return ""
    

def handle_prediction_result(result: dict):
    """
    Memutuskan apakah akan menampilkan form konfirmasi atau langsung menyimpan ke DB
    berdasarkan st.session_state.confirmation_enabled.
    """
    if st.session_state.confirmation_enabled:
        # Alur LAMA: Tampilkan form konfirmasi
        st.session_state.prediction = result
    else:
        # Alur BARU: Langsung simpan ke database
        # Pastikan amount tidak None sebelum konversi ke float
        amount_val = result.get('amount')
        if amount_val is None:
            amount_val = 0.0
            
        add_transaction_to_db(
            src=result['source'],
            text=result.get('raw_text', ''),
            label=result['label'],
            amount=float(amount_val),
            description=result['description']
        )
        st.success(f"Transaksi '{result.get('description', 'tanpa deskripsi')}' berhasil langsung disimpan!")
        st.session_state.prediction = None # Pastikan form tidak muncul
    
    st.rerun()
    

# Inisialisasi Aplikasi
init_db()

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if 'confirmation_enabled' not in st.session_state:
    st.session_state.confirmation_enabled = True 


# UI FiceAI di website
st.title("FiceAI — Organized Ur Finances")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.session_state.confirmation_enabled = st.toggle(
        "Aktifkan Form Konfirmasi",
        value=st.session_state.confirmation_enabled,
        help="Jika aktif, Kamu perlu memeriksa setiap transaksi sebelum disimpan. Jika tidak, transaksi akan langsung ditambahkan."
    )

st.markdown("---") #memperlihatkan jumlah saldo berdasarkan history laporan
balance = calculate_current_balance()
if balance >= 0:
    st.metric(label="Jumlah Uang Kamu Sekarang", value=f"Rp {balance:,.0f}")
else:
    st.metric(label="Oh no! Kamu Memiliki Hutang Sebesar", value=f"Rp {abs(balance):,.0f}", delta_color="inverse")
st.markdown("---")


with st.expander("ℹ️ How to use", expanded=False):
    st.write(
        "- **Input**: Use the 'Text', 'Voice', or 'Photo' tab to input a transaction.\n"
        "- **Confirm**: After processing, a confirmation form will appear. Correct the details if needed.\n"
        "- **Save**: Click 'Save Transaction' to add it to the history.\n"
        "- **Manage**: In the 'History' tab, you can delete specific entries or clear all data."
    )

if not (vectorizer and classifier):
    st.error("Model is not ready. Make sure the vectorizer & classifier files exist.")
    st.stop()
    
tab_text, tab_voice, tab_photo, tab_history = st.tabs(
    ["Text", "Voice", "Photo", "History"]
)

# bagian UI text
with tab_text:
    t = st.text_area("Masukkan keterangan transaksi", placeholder="ex: hasil jualan hari ini 2jt/makan hari ini 60rb", key="text_input")
    if st.button("Process Text", use_container_width=True):
        if not t.strip():
            st.warning("Text is empty.")
        else:
            with st.spinner("Analyzing..."):
                res = predict_sentence(t)
                if res["ok"]:
                    handle_prediction_result({**res, "source": "text"})
                else:
                    st.error(res.get("error"))
                    st.session_state.prediction = None

# bagian UI voice

with tab_voice:
    st.write("Anda bisa merekam suara langsung atau mengunggah file audio.")
    
    mic_audio = st.audio_input("Rekam transaksi Anda di sini:")
    st.write("--- Atau ---")
    file_audio = st.file_uploader("Unggah file audio", type=["wav", "mp3", "m4a", "ogg"])

    if st.button("Proses Suara", use_container_width=True):
        audio_data = None
        filename = "audio_input.wav"

        if mic_audio:
            audio_data = mic_audio.getvalue() 
        elif file_audio:
            audio_data = file_audio.getvalue()
            filename = file_audio.name

        if not audio_data:
            st.warning("Tidak ada input suara yang diberikan.")
        else:
            with st.spinner("Mentranskripsi dan menganalisis suara... 🎤"):
                text = stt_transcribe_audio_bytes(audio_data, filename)
                
                if not text:
                    st.error("Transkripsi gagal. Coba lagi dengan suara yang lebih jelas.")
                    st.session_state.prediction = None
                else:
                    st.info(f"**Teks Dikenali:** {text}")
                    res = predict_sentence(text)
                    if res["ok"]:
                        handle_prediction_result({**res, "source": "voice"})
                    else:
                        st.error(res.get("error"))
                        st.session_state.prediction = None

# mengkonfirmasi ulang prediksi dari model, fitur ini bisa di nonaktifkan dan di aktifkan sesuai kemauan user
if st.session_state.prediction:
    st.markdown("---")
    st.subheader("Confirm or Correct Transaction")
    
    pred = st.session_state.prediction
    
    with st.form(key="confirmation_form"):
        st.write(f"**Original Input Source:** {pred.get('source', 'N/A').capitalize()}")
        if pred.get('raw_text'):
             st.write(f"**Recognized Text:** {pred['raw_text']}")
        
        labels = ["pengeluaran", "pemasukan"]
        predicted_index = labels.index(pred['label']) if pred['label'] in labels else 0
        corrected_label = st.radio(
            "Category",
            options=labels,
            index=predicted_index,
            horizontal=True,
        )

        corrected_amount = st.number_input(
            "Amount (Rp)",
            value=float(pred.get('amount', 0.0) if pred.get('amount') is not None else 0.0),
            step=1000.0,
            format="%.2f"
        )
        
        corrected_desc = st.text_input("Description", value=pred.get('description', ''))
        
        submitted = st.form_submit_button("Save Transaction", use_container_width=True, type="primary")

        if submitted:
            add_transaction_to_db(
                src=pred['source'],
                text=pred.get('raw_text', 'Input from ' + pred['source']),
                label=corrected_label,
                amount=corrected_amount,
                description=corrected_desc
            )
            st.success("Transaction successfully saved!")
            st.session_state.prediction = None
            st.rerun()

# bagian UI photo
with tab_photo:
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("Terjadi masalah pada sistem kami, mohon tunggu sebentar...")
    else:
        API_KEY = st.secrets["GEMINI_API_KEY"]
        uploaded_file = st.file_uploader(
            "Upload gambar struk atau bukti transaksi", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Gambar yang di-upload", use_container_width=True)
            
            if st.button("Proses Gambar", use_container_width=True):
                with st.spinner("Menganalisis struk... 🤖"):
                    image_bytes = uploaded_file.getvalue()
                    result_dict = analyze_receipt_with_gemini(API_KEY, image_bytes)
                    
                    if result_dict:
                        # Siapkan data prediksi
                        prediction_data = {
                            "source": "photo",
                            "label": "pengeluaran", # asumsi setiap struk merupakan pengeluaran pengguna
                            "amount": result_dict.get("total_belanja", 0.0),
                            "description": result_dict.get("nama_merchant", "Transaksi dari foto"),
                            "raw_text": result_dict.get("teks_lengkap", "Teks tidak terbaca")
                        }
                        # Panggil handler agar toggle berfungsi
                        handle_prediction_result(prediction_data)
                    else:
                        st.error("Gagal menganalisis gambar. Coba gambar yang lebih jelas.")
                        st.session_state.prediction = None

# bagian UI history
with tab_history:
    st.header("Saved Transaction History")
    history_df = fetch_history()
    
    if history_df.empty:
        st.info("There is no transaction history")
    else:
        history_df_with_selection = history_df.copy()
        history_df_with_selection.insert(0, "select", False)
        
        st.write("You can check the rows you want to delete and click the delete button below.")
        
        edited_df = st.data_editor(
            history_df_with_selection,
            hide_index=True,
            column_config={"select": st.column_config.CheckboxColumn(required=True)},
            disabled=history_df.columns
        )
        
        selected_rows = edited_df[edited_df.select]
        selected_ids = selected_rows['id'].tolist()

        if st.button(f"Delete ({len(selected_ids)}) Selected Transactions", use_container_width=True):
            if selected_ids:
                delete_transactions_by_ids(selected_ids)
                st.success(f"{len(selected_ids)} transaction(s) have been deleted.")
                st.rerun()
            else:
                st.warning("No transactions selected for deletion.")

        st.download_button(
            label="Export to Excel",
            data=to_excel(history_df),
            file_name=f"finance_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        with st.expander("⚠️ Danger Zone"):
            if st.button("DELETE ALL HISTORY", use_container_width=True, type="primary"):
                delete_all_transactions()
                st.success("All transaction history has been cleared.")
                st.rerun()