import streamlit as st

_CSS = """
<style>
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    padding: 12px 14px;
}
</style>
"""

def apply_styles():
    """Inject the shared CSS. Call once near the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)

def render_risk_card(probability: float, threshold: float, label: str):
    """Show the churn result as a card with a probability bar and a threshold marker."""
    is_churn = label == "Churn"
    accent = "#DC2626" if is_churn else "#16A34A"
    tint = "#FEF2F2" if is_churn else "#F0FDF4"
    headline = "Likely to churn" if is_churn else "Not likely to churn"
    relation = "at or above" if is_churn else "below"

    p = max(0.0, min(probability, 1.0)) * 100
    t = max(0.0, min(threshold, 1.0)) * 100
    label_left = min(max(t, 10), 90)  # keeps the threshold label inside the card

    # Built as one unindented string: leading spaces or blank lines would make
    # Streamlit's markdown renderer treat the HTML as a code block.
    html = (
        f'<div style="background:{tint};border:1px solid {accent}33;'
        f'border-left:6px solid {accent};border-radius:12px;'
        f'padding:20px 24px;margin:8px 0 16px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'<div style="font-size:0.75rem;letter-spacing:0.08em;color:#64748B;">CHURN PROBABILITY</div>'
        f'<div style="font-size:2.4rem;font-weight:700;color:{accent};line-height:1.1;">{p:.1f}%</div>'
        f'</div>'
        f'<div style="background:{accent};color:#FFFFFF;font-weight:600;'
        f'padding:6px 16px;border-radius:999px;">{headline}</div>'
        f'</div>'
        f'<div style="position:relative;margin-top:18px;padding-top:4px;">'
        f'<div style="height:14px;background:#E2E8F0;border-radius:999px;overflow:hidden;">'
        f'<div style="width:{p:.2f}%;height:100%;background:{accent};"></div>'
        f'</div>'
        f'<div style="position:absolute;left:{t:.2f}%;top:0;height:22px;width:2px;background:#0F172A;"></div>'
        f'</div>'
        f'<div style="position:relative;height:22px;margin-top:6px;">'
        f'<span style="position:absolute;left:{label_left:.2f}%;transform:translateX(-50%);'
        f'font-size:0.8rem;color:#475569;white-space:nowrap;">Decision threshold {t:.1f}%</span>'
        f'</div>'
        f'<div style="font-size:0.85rem;color:#475569;margin-top:8px;">'
        f'The probability ({p:.1f}%) is {relation} the saved threshold ({t:.1f}%).'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)