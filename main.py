import streamlit as st
import pandas as pd
import numpy as np

# -------------------------
# 1. DATA LOADING
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("tableau_ready_colleges.csv")
    # if you don't have cluster labels in the file yet,
    # you can just use the numeric "Clusters" column.
    if "cluster_label" not in df.columns and "Clusters" in df.columns:
        df["cluster_label"] = "Cluster " + df["Clusters"].astype(str)
    # fallback: synthesize cluster labels from available score columns
    if "cluster_label" not in df.columns:
        eq = df.get("equity_score")
        val = df.get("value_equity_score", df.get("earnings_score"))
        # define thresholds using quantiles if scores look continuous; otherwise defaults
        def get_thresh(series, default=0.7):
            if series is None:
                return default
            try:
                s = pd.to_numeric(series, errors="coerce")
                if s.notna().sum() >= 10:
                    return float(s.quantile(0.7))
            except Exception:
                pass
            return default

        eq_t = get_thresh(eq, 0.7)
        val_t = get_thresh(val, 0.7)

        def assign_cluster(row):
            eq_v = pd.to_numeric(row.get("equity_score", np.nan), errors="coerce")
            val_v = pd.to_numeric(row.get("value_equity_score", row.get("earnings_score", np.nan)), errors="coerce")
            if pd.notna(eq_v) and pd.notna(val_v):
                if eq_v >= eq_t and val_v >= val_t:
                    return "High Value & High Equity"
                if val_v >= val_t and eq_v < eq_t:
                    return "High Value"
                if eq_v >= eq_t and val_v < val_t:
                    return "High Equity"
                return "Developing"
            # if missing, bucket by net_price if available
            price = pd.to_numeric(row.get("net_price", np.nan), errors="coerce")
            if pd.notna(price):
                return "Budget Friendly" if price <= pd.to_numeric(df.get("net_price", np.nan), errors="coerce").median() else "Premium"
            return "Uncategorized"

        df["cluster_label"] = df.apply(assign_cluster, axis=1)
    return df

df = load_data()

# -------------------------
# 2. HELPER: SCORING
# -------------------------
def build_weights(user_prefs):
    """
    user_prefs is a dict with booleans + sliders from the chat.
    We'll return a normalized dict of weights for each score field.
    """
    # base weights — your "default" social mobility perspective
    w = {
        "earnings_score":       0.20,
        "debt_score":           0.15,
        "net_price_score":      0.20,
        "aff_gap_score":        0.15,
        "aff_gap_parent_score": 0.05,
        "equity_score":         0.25,
    }

    # If user is URM, push equity higher
    if user_prefs["is_urm"]:
        w["equity_score"] += 0.10

    # If low-income, push affordability / debt higher
    if user_prefs["is_low_income"]:
        w["net_price_score"] += 0.10
        w["aff_gap_score"]   += 0.05
        w["debt_score"]      += 0.05

    # If student-parent, prioritize parent affordability
    if user_prefs["is_parent"]:
        w["aff_gap_parent_score"] += 0.10

    # Preference sliders (0-1 from user)
    w["earnings_score"]       += user_prefs["earnings_pref"] * 0.10
    w["debt_score"]           += user_prefs["debt_pref"] * 0.10
    w["equity_score"]         += user_prefs["equity_pref"] * 0.10
    w["net_price_score"]      += user_prefs["afford_pref"] * 0.10
    w["aff_gap_score"]        += user_prefs["afford_pref"] * 0.05

    # normalize
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    return w


def score_colleges(df_filtered, weights):
    df_scored = df_filtered.copy()
    personal = np.zeros(len(df_scored))
    for k, w in weights.items():
        if k in df_scored.columns:
            personal += w * df_scored[k].fillna(df_scored[k].median())
    df_scored["personal_score"] = personal
    return df_scored


# -------------------------
# 3. CHATBOT-STYLE UI
# -------------------------
st.set_page_config(page_title="Equity Path College Guide", layout="wide")

# left: chat, right: recommendations
chat_col, rec_col = st.columns([1, 2])

if "stage" not in st.session_state:
    st.session_state.stage = 0
    st.session_state.user_info = {}
    st.session_state.messages = []

def chat_message(role, text):
    with chat_col:
        if role == "bot":
            with st.chat_message("assistant", avatar="🧭"):
                st.markdown(text)
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(text)

# initial bot greeting
if st.session_state.stage == 0 and not st.session_state.messages:
    chat_message("bot",
        "Hi, I’m **Ava**, your college guide. I focus on students from underrepresented backgrounds and social mobility.\n\n"
        "I’ll ask a few questions about you, then I’ll recommend colleges in different clusters (like *High Value & High Equity*)."
    )
    st.session_state.stage = 1

# ------------- STAGE 1: BACKGROUND -------------
if st.session_state.stage == 1:
    with chat_col:
        with st.chat_message("assistant", avatar="🧭"):
            st.markdown("First, tell me a bit about yourself.")

            ethnicity = st.multiselect(
                "Which of these describe you? (it's okay to skip)",
                ["Black / African American", "Latino / Hispanic", "Native / Indigenous",
                 "Asian / Pacific Islander", "White", "Middle Eastern / North African",
                 "Another identity", "Prefer not to say"],
            )
            low_income = st.selectbox(
                "Would you describe your family as low-income / Pell-eligible?",
                ["Prefer not to say", "Yes", "No"],
            )
            first_gen = st.selectbox(
                "Are you a first-generation college student (first in your family to go to college)?",
                ["Prefer not to say", "Yes", "No"],
            )
            parent = st.selectbox(
                "Are you a student-parent (caring for children while in school)?",
                ["No", "Yes"],
            )
            if st.button("Next ▶"):
                st.session_state.user_info["ethnicity"] = ethnicity
                st.session_state.user_info["low_income"] = low_income
                st.session_state.user_info["first_gen"] = first_gen
                st.session_state.user_info["parent"] = (parent == "Yes")
                st.session_state.stage = 2

# ------------- STAGE 2: LOCATION + BUDGET -------------
if st.session_state.stage == 2:
    with chat_col:
        with st.chat_message("assistant", avatar="🧭"):
            st.markdown("Thanks! Now let's talk about where you might want to study and what you can afford.")

            states = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
            pref_states = st.multiselect(
                "Any states you’re especially interested in? (leave empty for 'anywhere')",
                states,
            )
            max_price = st.number_input(
                "What’s the highest **net price per year** you’re comfortable with? (you can change this later)",
                min_value=0,
                value=int(df["net_price"].median()) if "net_price" in df.columns else 30000,
                step=1000,
            )

            st.markdown("What matters most to you? Slide higher means more important.")
            earnings_pref = st.slider("Long-run earnings after college", 0.0, 1.0, 0.5)
            debt_pref = st.slider("Keeping my debt low", 0.0, 1.0, 0.7)
            equity_pref = st.slider("Colleges where students like me **actually graduate**", 0.0, 1.0, 0.8)
            afford_pref = st.slider("Overall affordability (net price, working while studying)", 0.0, 1.0, 0.7)

            if st.button("Show my matches 🎓"):
                st.session_state.user_info.update({
                    "pref_states": pref_states,
                    "max_price": max_price,
                    "earnings_pref": earnings_pref,
                    "debt_pref": debt_pref,
                    "equity_pref": equity_pref,
                    "afford_pref": afford_pref,
                })
                st.session_state.stage = 3

# ------------- STAGE 3: RECOMMENDATIONS -------------
if st.session_state.stage >= 3:
    u = st.session_state.user_info

    is_urm = any(e in u["ethnicity"] for e in [
        "Black / African American", "Latino / Hispanic", "Native / Indigenous"
    ])
    is_low_income = (u.get("low_income") == "Yes")
    is_parent = u.get("parent", False)

    prefs = {
        "is_urm": is_urm,
        "is_low_income": is_low_income,
        "is_parent": is_parent,
        "earnings_pref": u.get("earnings_pref", 0.5),
        "debt_pref": u.get("debt_pref", 0.5),
        "equity_pref": u.get("equity_pref", 0.5),
        "afford_pref": u.get("afford_pref", 0.5),
    }

    weights = build_weights(prefs)

    # filter
    df_filtered = df.copy()
    if u.get("pref_states"):
        df_filtered = df_filtered[df_filtered["state"].isin(u["pref_states"])]
    df_filtered = df_filtered[df_filtered["net_price"] <= u.get("max_price", 1e9)]

    df_scored = score_colleges(df_filtered, weights)

    # deduplicate institutions: keep the best-scoring row per institution
    def make_inst_key(df_in):
        if "Unit ID" in df_in.columns:
            return df_in["Unit ID"].astype(str)
        name_col = "Institution Name" if "Institution Name" in df_in.columns else ("Institution Name_x" if "Institution Name_x" in df_in.columns else ("Institution Name_y" if "Institution Name_y" in df_in.columns else None))
        state_col = "state" if "state" in df_in.columns else None
        if name_col is not None and state_col is not None:
            return df_in[name_col].astype(str) + " | " + df_in[state_col].astype(str)
        if name_col is not None:
            return df_in[name_col].astype(str)
        return pd.Series(np.arange(len(df_in)), index=df_in.index).astype(str)

    df_scored["__inst_key"] = make_inst_key(df_scored)
    df_scored = (
        df_scored
        .sort_values("personal_score", ascending=False)
        .drop_duplicates(subset="__inst_key", keep="first")
    )

    # group by cluster_label and pick top N per cluster
    N_PER_CLUSTER = 4
    recs = (
        df_scored
        .sort_values("personal_score", ascending=False)
        .groupby("cluster_label", group_keys=False)
        .head(N_PER_CLUSTER)
    )

    # bot summary message
    chat_text = (
        "Based on what you told me, I’m prioritizing:\n"
        f"- **Equity score** weight: `{weights['equity_score']:.2f}`\n"
        f"- **Affordability (net price + gap)** weight: `{weights['net_price_score'] + weights['aff_gap_score']:.2f}`\n"
        f"- **Debt** weight: `{weights['debt_score']:.2f}`\n"
        f"- **Earnings** weight: `{weights['earnings_score']:.2f}`\n\n"
        "Here are colleges grouped by **cluster type**. Click a card for more details."
    )
    chat_message("bot", chat_text)

    # -------------------------
    # 4. PROFILE CARDS
    # -------------------------
    with rec_col:
        st.header("Recommended colleges for you")

        for cluster_name, group in recs.groupby("cluster_label"):
            st.subheader(cluster_name)
            card_cols = st.columns(2)
            group = group.sort_values("personal_score", ascending=False)

            for i, (_, row) in enumerate(group.iterrows()):
                col = card_cols[i % 2]
                with col:
                    equity_badge = "⭐ High Equity" if row.get("equity_score", 0) >= 0.75 else ""
                    debt_badge = "💳 Low Debt" if row.get("debt_score", 0) >= 0.75 else ""
                    mobility_badge = "🚀 Mobility" if (row.get("earnings_score",0) >= 0.7 and row.get("net_price_score",0) >= 0.7) else ""

                    inst_name = row.get("Institution Name", row.get("Institution Name_x", row.get("Institution Name_y", "Unknown Institution")))
                    inst_state = row.get("state", "")
                    with st.expander(f"{inst_name} ({inst_state})"):
                        st.markdown(
                            f"**Cluster:** {cluster_name}  \n"
                            f"**Badges:** {equity_badge} {debt_badge} {mobility_badge}"
                        )
                        st.markdown("---")
                        st.markdown(
                            f"- **Personal match score:** `{row['personal_score']:.3f}`  \n"
                            f"- **Net price (est.):** `${row['net_price']:,.0f}` per year  \n"
                            f"- **Median earnings (10 yrs):** `${row['median_earn_10yr']:,.0f}`  \n"
                            f"- **Median debt:** `${row['median_debt']:,.0f}`  "
                        )

                        # Equity details if available
                        eq = row.get("equity_score", np.nan)
                        if not np.isnan(eq):
                            st.markdown(
                                f"**Equity score:** `{eq:.2f}` (higher = smaller graduation gaps)"
                            )

                        # You can add more charts here later (grad gaps by group, etc.)

        st.caption("Note: These recommendations use public data and can’t capture everything about fit (campus culture, support programs, etc.). Use them as a starting point for deeper research.")
