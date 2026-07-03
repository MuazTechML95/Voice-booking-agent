"""
streamlit_app.py
------------------
AI-Powered Voice Appointment Booking Agent — main Streamlit entry point.
Run with:
    streamlit run streamlit_app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from modules.utils import load_config, get_categories_for_business, get_logger
from modules.conversation_manager import ConversationManager
from modules.appointment_manager import AppointmentManager
from modules.speech_to_text import transcribe_audio
from modules.text_to_speech import synthesize_speech

load_dotenv()
logger = get_logger(__name__)

st.set_page_config(page_title="Voice Appointment Booking Agent", page_icon="📅", layout="wide")

# --------------------------------------------------------------------------- #
# Custom CSS
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark purple background */
.stApp {
    background: #1a1a2e !important;
}

/* Main content area */
.main .block-container {
    background: #f8f9ff;
    border-radius: 20px;
    padding: 2rem 2.5rem !important;
    margin: 1rem;
    min-height: calc(100vh - 2rem);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #16213e !important;
}
[data-testid="stSidebar"] > div {
    background: #16213e !important;
    padding: 1.5rem 1rem;
}
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    padding-bottom: 0.75rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.15) !important;
    margin-bottom: 1rem !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #a5b4fc !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {
    color: #818cf8 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Page header gradient card */
.page-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(79,70,229,0.35);
}
.page-header h2 {
    color: #ffffff !important;
    margin: 0 !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.page-header p {
    color: rgba(255,255,255,0.85) !important;
    margin: 0.4rem 0 0 0 !important;
    font-size: 0.95rem !important;
}

/* Headings */
h1, h2, h3 {
    color: #1e1b4b !important;
    font-weight: 700 !important;
}

/* Chat messages - FORCE VISIBLE */
[data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
    color: #1e1b4b !important;
    border: 1px solid #e0e7ff !important;
}
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.25rem 0 !important;
}
/* User chat bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
}

/* Chat container box */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: #f0f4ff;
    border-radius: 16px;
    padding: 1rem;
    border: 2px solid #c7d2fe;
}

/* Section cards */
.section-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #c7d2fe;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08);
    margin-bottom: 1rem;
}

/* Summary card */
.summary-card {
    background: linear-gradient(135deg, #eef2ff, #ede9fe);
    border-radius: 16px;
    padding: 1.5rem;
    border: 2px solid #a5b4fc;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(99,102,241,0.1);
}

/* Badge */
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(79,70,229,0.45) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #ffffff !important;
    color: #4f46e5 !important;
    border: 2px solid #6366f1 !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: #eef2ff !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #c7d2fe !important;
    background: #ffffff !important;
    color: #1e1b4b !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    border: 1px solid #c7d2fe !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08) !important;
}
[data-testid="stMetricValue"] {
    color: #4f46e5 !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 2px solid #c7d2fe !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #e0e7ff !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #4338ca !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #4f46e5 !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.2) !important;
}

/* Audio input */
[data-testid="stAudioInput"] {
    background: #eef2ff !important;
    border-radius: 16px !important;
    border: 2px dashed #818cf8 !important;
    padding: 1rem !important;
}

/* Alerts */
.stSuccess > div {
    background: #d1fae5 !important;
    border: 1px solid #6ee7b7 !important;
    border-radius: 12px !important;
    color: #065f46 !important;
}
.stWarning > div {
    background: #fef3c7 !important;
    border: 1px solid #fcd34d !important;
    border-radius: 12px !important;
    color: #92400e !important;
}
.stError > div {
    background: #fee2e2 !important;
    border: 1px solid #fca5a5 !important;
    border-radius: 12px !important;
    color: #991b1b !important;
}
.stInfo > div {
    background: #e0e7ff !important;
    border: 1px solid #a5b4fc !important;
    border-radius: 12px !important;
    color: #3730a3 !important;
}

/* Form */
[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 16px !important;
    padding: 1.25rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #eef2ff; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #818cf8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* Caption text */
.stCaption { color: #6b7280 !important; font-size: 0.85rem !important; }

/* Divider */
hr { border-color: #c7d2fe !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Config & managers
# --------------------------------------------------------------------------- #
try:
    CONFIG = load_config()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load configuration: {exc}")
    st.stop()

APPOINTMENT_MANAGER = AppointmentManager(CONFIG)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = None
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "voice_reply_enabled" not in st.session_state:
    st.session_state.voice_reply_enabled = True
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def start_new_conversation(business_type: str):
    manager = ConversationManager(CONFIG, business_type)
    st.session_state.conversation_manager = manager
    greeting = manager.greeting_message()
    st.session_state.messages = [{"role": "assistant", "text": greeting}]
    st.session_state.last_audio_id = None
    speak_if_enabled(greeting)


def push_message(role: str, text: str):
    st.session_state.messages.append({"role": role, "text": text})


def speak_if_enabled(text: str):
    if not st.session_state.voice_reply_enabled:
        st.session_state.pending_audio = None
        return
    audio_path = synthesize_speech(text)
    st.session_state.pending_audio = audio_path


def process_user_text(user_text: str):
    manager: ConversationManager = st.session_state.conversation_manager
    if manager is None:
        return
    push_message("user", user_text)
    result = manager.handle_message(user_text)
    if result["done"]:
        try:
            appointment_id = APPOINTMENT_MANAGER.book_appointment(result["appointment"])
            success_text = manager.render_message("success_message", appointment_id=appointment_id)
        except Exception as exc:
            logger.error("Failed to save appointment: %s", exc)
            success_text = "Sorry, something went wrong while saving your appointment. Please try again."
        push_message("assistant", success_text)
        speak_if_enabled(success_text)
    else:
        push_message("assistant", result["reply"])
        speak_if_enabled(result["reply"])


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown("# 📅 Booking Agent")

business_types = CONFIG.get("business_types", [])
selected_business_type = st.sidebar.selectbox("Business Type", business_types, index=0)

st.session_state.voice_reply_enabled = st.sidebar.checkbox(
    "🔊 Speak responses out loud (Text-to-Speech)",
    value=st.session_state.voice_reply_enabled
)

if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
    st.sidebar.warning("⚠️ No API key — running in rule-based mode.")

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Start New Booking", use_container_width=True):
    start_new_conversation(selected_business_type)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["💬 Book Appointment", "📊 History & Dashboard", "🛠️ Manage Appointment"]
)

# Auto-start
if st.session_state.conversation_manager is None:
    start_new_conversation(selected_business_type)
elif st.session_state.conversation_manager.business_type != selected_business_type:
    start_new_conversation(selected_business_type)


# --------------------------------------------------------------------------- #
# Page 1 — Booking
# --------------------------------------------------------------------------- #
def render_booking_page():
    st.markdown(f"""
    <div class="page-header">
        <h2>🎙️ Voice Appointment Booking</h2>
        <p>Business: <strong>{selected_business_type}</strong> &nbsp;•&nbsp; 💡 Ask about hours, services, or pricing anytime</p>
    </div>
    """, unsafe_allow_html=True)

    manager: ConversationManager = st.session_state.conversation_manager

    # Chat transcript
    st.markdown("### 💬 Conversation")
    chat_box = st.container(height=350, border=True)
    with chat_box:
        if not st.session_state.messages:
            st.info("Start a conversation by speaking or typing below.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["text"])

    st.markdown("")

    # Play the agent's latest voice reply (if any) with autoplay
    if st.session_state.pending_audio:
        st.audio(st.session_state.pending_audio, format="audio/wav", autoplay=True)

    st.markdown("")
    if manager.state == manager.STATE_CONFIRMING:
        render_summary_card(manager.answers)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Appointment", type="primary", use_container_width=True):
                process_user_text("yes")
                st.rerun()
        with col2:
            if st.button("✏️ Edit Details", use_container_width=True):
                process_user_text("no")
                st.rerun()

    elif manager.state in (manager.STATE_DONE, manager.STATE_CANCELLED):
        if manager.state == manager.STATE_DONE:
            st.success("🎉 Appointment booked successfully!")
        else:
            st.info("❌ Booking cancelled. Click 'Start New Booking' to begin again.")

    else:
        col_voice, col_text = st.columns(2, gap="large")

        with col_voice:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### 🎤 Voice Input")
            st.caption("Tap the mic and speak your response")
            audio_value = st.audio_input("Record your response")
            if audio_value is not None:
                audio_bytes = audio_value.getvalue()
                audio_id = hash(audio_bytes)
                if audio_id != st.session_state.last_audio_id:
                    st.session_state.last_audio_id = audio_id
                    with st.spinner("🔄 Transcribing..."):
                        ok, text_or_error = transcribe_audio(audio_bytes)
                    if ok:
                        process_user_text(text_or_error)
                        st.rerun()
                    else:
                        st.error(text_or_error)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_text:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### ⌨️ Type Response")
            st.caption("Prefer typing? Use the box below")
            with st.form(key="text_input_form", clear_on_submit=True):
                text_value = st.text_input(
                    "Your message",
                    placeholder="Type here and press Enter...",
                    label_visibility="collapsed"
                )
                submitted = st.form_submit_button("Send Message →", use_container_width=True, type="primary")
            if submitted:
                if text_value.strip() == "":
                    st.warning("Please enter a message.")
                else:
                    process_user_text(text_value)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def render_summary_card(answers: dict):
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Appointment Summary")
    st.markdown('<span class="badge">Please Review & Confirm</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**🏢 Business:** {answers.get('business_type', '-')}")
        st.markdown(f"**📂 Category:** {answers.get('category', '-')}")
        st.markdown(f"**👤 Name:** {answers.get('full_name', '-')}")
    with c2:
        st.markdown(f"**📞 Phone:** {answers.get('phone', '-')}")
        st.markdown(f"**📅 Date:** {answers.get('appointment_date', '-')}")
        st.markdown(f"**🕐 Time:** {answers.get('appointment_time', '-')}")
    st.markdown(f"**📝 Purpose:** {answers.get('purpose', '-')}")
    st.caption("✅ Review the details above and confirm to book.")
    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page 2 — History & Dashboard
# --------------------------------------------------------------------------- #
def render_history_page():
    st.markdown("""
    <div class="page-header">
        <h2>📊 Appointment History & Dashboard</h2>
        <p>View all bookings, metrics, and appointment records</p>
    </div>
    """, unsafe_allow_html=True)

    scope = st.radio(
        "Show appointments for:",
        ["All Businesses", selected_business_type],
        horizontal=True
    )
    filter_type = None if scope == "All Businesses" else selected_business_type

    try:
        metrics = APPOINTMENT_MANAGER.get_metrics(filter_type)
        history = APPOINTMENT_MANAGER.get_history(filter_type)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        return

    st.markdown("")
    m1, m2, m3 = st.columns(3)
    m1.metric("📋 Total Appointments", metrics["total"])
    m2.metric("✅ Confirmed", metrics["confirmed"])
    m3.metric("❌ Cancelled", metrics["cancelled"])

    st.markdown("---")
    st.markdown("### 📄 All Appointments")
    if history:
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("📭 No appointments yet. Book one from the Booking page!")


# --------------------------------------------------------------------------- #
# Page 3 — Manage
# --------------------------------------------------------------------------- #
def render_manage_page():
    st.markdown("""
    <div class="page-header">
        <h2>🛠️ Manage Appointments</h2>
        <p>Reschedule or cancel an existing appointment by ID</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Appointment ID")
    appointment_id = st.number_input(
        "Enter the Appointment ID (find it in History page)",
        min_value=1, step=1,
        label_visibility="visible"
    )

    st.markdown("")
    tab1, tab2 = st.tabs(["📅 Reschedule Appointment", "🗑️ Cancel Appointment"])

    with tab1:
        st.markdown("#### Enter New Schedule")
        col1, col2 = st.columns(2)
        with col1:
            new_date = st.text_input(
                "📅 New Date",
                placeholder="e.g. 2026-07-15 or 15th July",
                key="resched_date"
            )
        with col2:
            new_time = st.text_input(
                "🕐 New Time",
                placeholder="e.g. 3 PM or 15:00",
                key="resched_time"
            )
        st.markdown("")
        if st.button("🔄 Reschedule Appointment", type="primary", use_container_width=True):
            if not new_date.strip() or not new_time.strip():
                st.warning("⚠️ Please provide both new date and new time.")
            else:
                ok, result = APPOINTMENT_MANAGER.reschedule_appointment(
                    int(appointment_id), new_date, new_time
                )
                if ok:
                    st.success(
                        f"✅ Appointment #{int(appointment_id)} rescheduled to "
                        f"**{result['appointment_date']}** at **{result['appointment_time']}**."
                    )
                else:
                    st.error(f"❌ {result}")

    with tab2:
        st.markdown("#### Cancel Appointment")
        st.warning(
            "⚠️ **Warning:** This action cannot be undone. "
            "The appointment will be permanently marked as CANCELLED."
        )
        st.markdown("")
        if st.button("🗑️ Cancel This Appointment", type="primary", use_container_width=True):
            ok, message = APPOINTMENT_MANAGER.cancel_appointment(int(appointment_id))
            if ok:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
if page == "💬 Book Appointment":
    render_booking_page()
elif page == "📊 History & Dashboard":
    render_history_page()
else:
    render_manage_page()