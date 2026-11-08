from __future__ import annotations

import streamlit as st

PANEL_W = "23rem"

BASE_CSS = """
/* ---------- page frame ---------- */
[data-testid="stHeader"] { background: transparent; }
.stMainBlockContainer { max-width: 50rem; padding: 4rem 1.5rem 8rem; }

/* ---------- sidebar: flat, left-aligned rows like Gemini ---------- */
[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; text-align: left; }
[data-testid="stSidebar"] .stButton > button > div,
[data-testid="stSidebar"] .stButton > button > div > span { justify-content: flex-start; width: 100%; text-align: left; }
[data-testid="stSidebar"] [data-testid="stPopoverButton"][kind="tertiary"] { padding-left: .35rem; padding-right: .35rem; }
/* icon-only popovers: hide the dropdown chevron */
[data-testid="stPopoverButton"][kind="tertiary"] div[aria-hidden="true"] { display: none; }
.st-key-new_chat button { border-radius: 999px; background: rgba(128,128,128,.16); border: none; }
.st-key-sidebar_user { position: sticky; bottom: 0; padding: .6rem 0 .2rem; background: inherit; }
.side-label { font-size: .78rem; letter-spacing: .02em; opacity: .6; margin: .9rem 0 .2rem .25rem; }
.brand { font-size: 1.25rem; font-weight: 600; margin: .2rem 0 .8rem .25rem; }

/* ---------- conversation ---------- */
[class*="st-key-user_msg_"] {
  margin-left: auto; width: fit-content; max-width: 82%; gap: 0;
  background: rgba(128,128,128,.18); border-radius: 22px; padding: .7rem 1.25rem; margin-top: 1.1rem;
}
/* Streamlit cancels paragraph spacing with a negative margin; zero both so the bubble isn't clipped */
[class*="st-key-user_msg_"] [data-testid="stMarkdownContainer"] { margin: 0 !important; }
[class*="st-key-user_msg_"] p { margin: 0 !important; line-height: 1.55; }
[class*="st-key-asst_msg_"] { margin: .4rem 0 .2rem; }
/* answers often start with big markdown headings: keep them in proportion */
[class*="st-key-asst_msg_"] h1 { font-size: 1.55rem; line-height: 1.3; padding: .4rem 0 .1rem; }
[class*="st-key-asst_msg_"] h2 { font-size: 1.3rem; line-height: 1.3; padding: .4rem 0 .1rem; }
[class*="st-key-asst_msg_"] h3 { font-size: 1.12rem; line-height: 1.3; padding: .3rem 0 .1rem; }
.st-key-user_actions, .st-key-asst_actions { gap: 0; margin-top: -.3rem; }
.st-key-user_actions button, .st-key-asst_actions button { opacity: .75; }

/* ---------- new-chat hero ---------- */
.st-key-hero { padding-top: 14vh; }
.hero-title { text-align: center; font-size: 2.6rem; font-weight: 400; line-height: 1.2; margin-bottom: .3rem; }
.hero-sub { text-align: center; opacity: .65; margin-bottom: 1.6rem; }
.st-key-suggestions { margin-top: 1.2rem; }
.st-key-suggestions button { justify-content: flex-start; text-align: left; }

/* ---------- composer ---------- */
[data-testid="stChatInput"] { max-width: 47rem; margin: 0 auto; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div { border-radius: 28px; }
[data-testid="stChatInput"] { max-width: 47rem; margin-left: auto; margin-right: auto; }
.st-key-stopbar {
  position: sticky; bottom: 1.1rem; z-index: 50; margin-top: 1rem;
  background: var(--secondary-background-color, #1e1f20);
  border: 1px solid rgba(128,128,128,.3); border-radius: 999px; padding: .35rem .45rem .35rem 1.3rem;
}
.st-key-stopbar p { margin: 0; opacity: .75; }

/* ---------- top bar ---------- */
.st-key-topbar { margin-bottom: .2rem; }
.st-key-topbar h4 { margin: 0; padding: 0; }
"""

PANEL_CSS = f"""
/* ---------- documents panel: docked on the right ---------- */
.st-key-docs_panel {{
  position: fixed; top: 3.6rem; right: 0; bottom: 0; width: {PANEL_W}; z-index: 90;
  overflow-y: auto; padding: 1rem 1.1rem 2rem;
  background: var(--secondary-background-color, #1e1f20);
  border-left: 1px solid rgba(128,128,128,.25);
}}
[data-testid="stMain"] {{ padding-right: {PANEL_W}; }}
"""


def inject(docs_open: bool) -> None:
    css = BASE_CSS + (PANEL_CSS if docs_open else "")
    st.html(f"<style>{css}</style>")
