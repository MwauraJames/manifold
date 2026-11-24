from __future__ import annotations

import streamlit as st

PANEL_W = "23rem"

BASE_CSS = """
/* ---------- page frame ---------- */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] [data-testid="stToolbar"] {
  opacity: 0.35;
  transition: opacity 0.18s ease;
}
[data-testid="stHeader"] [data-testid="stToolbar"]:hover { opacity: 1; }
.stMainBlockContainer { max-width: 50rem; padding: 3.5rem 1.5rem 2rem; }

/* ---------- sidebar: flat, left-aligned rows ---------- */
[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; text-align: left; }
[data-testid="stSidebar"] .stButton > button > div,
[data-testid="stSidebar"] .stButton > button > div > span { justify-content: flex-start; width: 100%; text-align: left; }
[data-testid="stSidebar"] [data-testid="stPopoverButton"][kind="tertiary"] { padding-left: .35rem; padding-right: .35rem; }
[data-testid="stPopoverButton"][kind="tertiary"] div[aria-hidden="true"] { display: none; }
.st-key-new_chat button { border-radius: 999px; background: rgba(128,128,128,.16); border: none; }
.st-key-sidebar_user { position: sticky; bottom: 0; padding: .6rem 0 .2rem; background: inherit; }
.side-label { font-size: .78rem; letter-spacing: .02em; opacity: .6; margin: .9rem 0 .2rem .25rem; }
.brand { font-size: 1.25rem; font-weight: 600; margin: .2rem 0 .8rem .25rem; }

/* ---------- top bar ---------- */
.st-key-topbar { margin-bottom: .4rem; }
.st-key-topbar h4 { margin: 0; padding: 0; font-weight: 500; letter-spacing: -0.01em; }
.st-key-docs_toggle button {
  border-radius: 10px;
  border: 1px solid rgba(128, 128, 128, .25);
  background: transparent;
  font-size: .86rem;
  transition: background .15s ease, border-color .15s ease;
}
.st-key-docs_toggle button:hover {
  background: rgba(128, 128, 128, .12);
  border-color: rgba(128, 128, 128, .4);
}

/* ---------- conversation ---------- */
[class*="st-key-user_msg_"] {
  margin-left: auto; width: fit-content; max-width: 82%; gap: 0;
  background: rgba(128,128,128,.16); border-radius: 20px; padding: .68rem 1.2rem; margin-top: 1.1rem;
}
[class*="st-key-user_msg_"] [data-testid="stMarkdownContainer"] { margin: 0 !important; }
[class*="st-key-user_msg_"] p { margin: 0 !important; line-height: 1.55; }
[class*="st-key-asst_msg_"] { margin: .45rem 0 .2rem; }
[class*="st-key-asst_msg_"] h1 { font-size: 1.55rem; line-height: 1.3; padding: .4rem 0 .1rem; }
[class*="st-key-asst_msg_"] h2 { font-size: 1.3rem; line-height: 1.3; padding: .4rem 0 .1rem; }
[class*="st-key-asst_msg_"] h3 { font-size: 1.12rem; line-height: 1.3; padding: .3rem 0 .1rem; }
.st-key-user_actions, .st-key-asst_actions { gap: 0; margin-top: -.3rem; }
.st-key-user_actions button, .st-key-asst_actions button { opacity: .65; }
.st-key-user_actions button:hover, .st-key-asst_actions button:hover { opacity: 1; }

/* ---------- new-chat hero ---------- */
.st-key-hero { padding-top: 13vh; }
.hero-title {
  text-align: center;
  font-size: 2.45rem;
  font-weight: 400;
  letter-spacing: -0.02em;
  line-height: 1.2;
  margin-bottom: .35rem;
}
.hero-sub {
  text-align: center;
  font-size: .95rem;
  opacity: .6;
  margin-bottom: 1.8rem;
}

/* ---------- starter suggestion pills ---------- */
.st-key-suggestions {
  margin-top: 1.1rem;
  gap: .5rem !important;
  flex-wrap: wrap;
}
.st-key-suggestions button {
  border-radius: 999px !important;
  border: 1px solid rgba(128, 128, 128, .24) !important;
  background: rgba(128, 128, 128, .06) !important;
  padding: .32rem .85rem !important;
  min-height: 2.1rem !important;
  font-size: .84rem !important;
  opacity: .88;
  transition: background .15s ease, border-color .15s ease, opacity .15s ease;
}
.st-key-suggestions button:hover {
  background: rgba(128, 128, 128, .16) !important;
  border-color: rgba(128, 128, 128, .42) !important;
  opacity: 1;
}

/* ---------- unified Claude-style composer card ---------- */
.st-key-composer_hero,
.st-key-composer_docked {
  max-width: 46rem;
  width: 100%;
  margin-left: auto;
  margin-right: auto;
  gap: .1rem !important;
  background: var(--secondary-background-color, #1e1f20);
  border: 1px solid rgba(128, 128, 128, .24);
  border-radius: 20px;
  padding: .35rem .55rem .45rem;
  box-shadow: 0 6px 24px rgba(0, 0, 0, .2);
  transition: border-color .18s ease, box-shadow .18s ease;
}
.st-key-composer_hero:focus-within,
.st-key-composer_docked:focus-within {
  border-color: rgba(128, 128, 128, .48);
  box-shadow: 0 8px 28px rgba(0, 0, 0, .28);
}

/* pin the docked composer & stop bar to the viewport bottom when scrolling up */
.stMainBlockContainer > [data-testid="stVerticalBlock"] > div:has(.st-key-composer_docked),
.stMainBlockContainer > [data-testid="stVerticalBlock"] > div:has(.st-key-stopbar) {
  position: sticky !important;
  bottom: 0 !important;
  z-index: 60;
  padding: .75rem 0 1rem;
  background: linear-gradient(
    to top,
    var(--background-color) 78%,
    transparent 100%
  );
}
.st-key-composer_docked {
  margin-top: .5rem;
}

/* strip st.chat_input's own box so it blends into the card */
.st-key-composer_hero [data-testid="stChatInput"],
.st-key-composer_docked [data-testid="stChatInput"] {
  max-width: 100%;
  margin: 0;
  background: transparent !important;
}
.st-key-composer_hero [data-testid="stChatInput"] > div,
.st-key-composer_docked [data-testid="stChatInput"] > div {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  padding-left: .2rem;
}
.st-key-composer_hero [data-testid="stChatInputTextArea"],
.st-key-composer_docked [data-testid="stChatInputTextArea"] {
  font-size: .96rem;
}

/* bottom toolbar inside composer: borderless ghost controls */
.st-key-composer_toolbar {
  padding: .15rem .25rem 0;
  margin-top: -.1rem;
}
.st-key-composer_toolbar [data-testid="stHorizontalBlock"] {
  gap: .35rem;
  align-items: center;
}
.st-key-composer_toolbar [data-baseweb="select"] > div {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  min-height: 2rem !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  font-size: .83rem !important;
  opacity: .84;
  transition: background .15s ease, opacity .15s ease;
}
.st-key-composer_toolbar [data-baseweb="select"] > div:hover {
  background: rgba(128, 128, 128, .15) !important;
  opacity: 1;
}
.st-key-composer_toolbar [data-baseweb="select"] input {
  font-size: .83rem !important;
}
.st-key-composer_add_docs button {
  border-radius: 10px !important;
  min-height: 2rem !important;
  padding: .2rem .45rem !important;
  opacity: .78;
}
.st-key-composer_add_docs button:hover {
  background: rgba(128, 128, 128, .15) !important;
  opacity: 1;
}

/* ---------- stop bar during streaming ---------- */
.st-key-stopbar {
  max-width: 46rem;
  margin: 0 auto;
  background: var(--secondary-background-color, #1e1f20);
  border: 1px solid rgba(128,128,128,.28);
  border-radius: 999px;
  padding: .35rem .45rem .35rem 1.3rem;
}
.st-key-stopbar p { margin: 0; opacity: .75; font-size: .9rem; }

/* ---------- inline citations & source drawer ---------- */
.cite-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  font-weight: 600;
  line-height: 1;
  padding: 0.14rem 0.42rem;
  margin: 0 0.12rem;
  border-radius: 999px;
  background: rgba(99, 102, 241, 0.18);
  color: #818cf8;
  border: 1px solid rgba(99, 102, 241, 0.35);
  cursor: help;
  vertical-align: super;
  text-decoration: none;
  transition: background 0.15s ease, transform 0.1s ease;
}
.cite-badge:hover {
  background: rgba(99, 102, 241, 0.32);
  transform: translateY(-1px);
}
.source-card {
  border: 1px solid rgba(128, 128, 128, 0.22);
  border-radius: 12px;
  padding: 0.65rem 0.85rem;
  margin-bottom: 0.55rem;
  background: rgba(128, 128, 128, 0.06);
}
.source-card.is-cited {
  border-left: 3px solid #6366f1;
}
.source-card-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.4rem;
  margin-bottom: 0.35rem;
  font-size: 0.83rem;
}
.source-card-title { font-weight: 600; }
.source-card-meta {
  font-size: 0.75rem;
  opacity: 0.7;
  display: flex;
  gap: 0.45rem;
  align-items: center;
}
.source-pill {
  padding: 0.1rem 0.45rem;
  border-radius: 999px;
  font-size: 0.7rem;
  background: rgba(128, 128, 128, 0.16);
}
.source-pill.cited-pill {
  background: rgba(99, 102, 241, 0.2);
  color: #818cf8;
  font-weight: 600;
}

/* ---------- compact document rows ---------- */
.doc-row-title {
  font-size: .88rem;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.35;
}
.doc-row-meta {
  font-size: .75rem;
  opacity: .65;
  display: flex;
  align-items: center;
  gap: .35rem;
  margin-top: .1rem;
}
.doc-dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 999px;
  flex-shrink: 0;
}
.doc-dot.ready { background: #34d399; }
.doc-dot.processing { background: #fbbf24; }
.doc-dot.queued { background: #9ca3af; }
.doc-dot.failed { background: #f87171; }
"""

PANEL_CSS = f"""
/* ---------- documents panel: docked on the right ---------- */
.st-key-docs_panel {{
  position: fixed; top: 3.6rem; right: 0; bottom: 0; width: {PANEL_W}; z-index: 90;
  overflow-y: auto; padding: 1rem 1.1rem 2rem;
  background: var(--secondary-background-color, #1e1f20);
  border-left: 1px solid rgba(128,128,128,.2);
}}
[data-testid="stMain"] {{ padding-right: {PANEL_W}; }}

/* subtle active state for the Documents toggle button */
.st-key-docs_toggle button {{
  background: rgba(138, 180, 248, .12);
  border-color: rgba(138, 180, 248, .42);
  color: var(--primary-color, #8ab4f8);
}}
"""


def inject(docs_open: bool) -> None:
    css = BASE_CSS + (PANEL_CSS if docs_open else "")
    st.html(f"<style>{css}</style>")