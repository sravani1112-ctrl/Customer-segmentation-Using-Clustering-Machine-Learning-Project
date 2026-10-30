import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ClusterIQ | Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROFESSIONAL DARK UI
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        linear-gradient(125deg, rgba(76, 101, 165, 0.14), transparent 42%),
        linear-gradient(315deg, rgba(54, 137, 119, 0.08), transparent 38%),
        #0b0d14;
    color: #f5f5f7;
}


/* Hide Streamlit header */

header {
    visibility: hidden;
}


/* Main container */

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 4rem;
    max-width: 1480px;
}


/* Navigation */

.navbar {
    padding: 5px 0 14px;
    margin-bottom: 30px;
    border-bottom: 1px solid rgba(255,255,255,0.09);
}


/* Logo */

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 48px;
}

.brand-mark {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    border-radius: 9px;
    color: #0b0d14;
    background: linear-gradient(140deg, #7bd8bd, #a4afff);
    font-size: 21px;
    font-weight: 700;
}

.logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px;
    font-weight: 800;
    color: white;
}

.logo span {
    color: #83d9bf;
}

div[role="radiogroup"] {
    justify-content: flex-end;
    gap: 3px;
}

div[role="radiogroup"] label {
    border-radius: 7px;
    padding: 7px 11px;
    color: #a5aab8;
}

div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.06);
    color: #ffffff;
}


/* Hero */

.hero-badge {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 6px;
    background: rgba(123,216,189,0.09);
    border: 1px solid rgba(123,216,189,0.2);
    color: #9be3cd;
    font-size: 12px;
    font-weight: 600;
    margin: 13px 0 18px;
}


.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 52px;
    line-height: 1.06;
    font-weight: 800;
    letter-spacing: 0;
    margin-bottom: 15px;
}


.gradient-text {
    background: linear-gradient(
        90deg,
        #83d9bf,
        #a4afff,
        #f1a0c6
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.hero-description {
    font-size: 16px;
    color: #a2a8b8;
    line-height: 1.7;
    max-width: 590px;
}

.visual-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin: 11px 2px 5px;
    color: #f0f1f6;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 600;
}

.visual-status {
    color: #83d9bf;
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 700;
}

.hero-note {
    color: #858c9c;
    font-size: 12px;
    margin-top: -4px;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 3px;
}

.hero-tag {
    padding: 6px 9px;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 6px;
    background: rgba(255,255,255,0.035);
    color: #c1c5d0;
    font-size: 11px;
}


/* Cards */

.metric-card {
    background: rgba(24, 27, 38, 0.78);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 19px;
    min-height: 125px;
}


.metric-label {
    color: #9ca3b5;
    font-size: 14px;
    margin-bottom: 9px;
}


.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
}


/* Sections */

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 750;
    margin-top: 43px;
    margin-bottom: 8px;
}


.section-description {
    color: #8f96aa;
    margin-bottom: 25px;
}


/* Workflow */

.workflow-card {
    background: rgba(24,27,38,0.7);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 19px;
    text-align: center;
}


.workflow-number {
    color: #83d9bf;
    font-weight: 800;
    font-size: 13px;
}


.workflow-title {
    font-weight: 700;
    margin-top: 7px;
}


/* Feature cards */

.feature-card {
    background: rgba(22,24,42,0.7);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 25px;
    height: 180px;
}


.feature-icon {
    font-size: 28px;
    margin-bottom: 12px;
}


.feature-title {
    font-weight: 700;
    font-size: 17px;
}


.feature-text {
    color: #8f96aa;
    font-size: 14px;
    line-height: 1.6;
}


/* Buttons */

.stButton > button {
    border-radius: 7px;
    border: 1px solid rgba(123,216,189,0.35);
    background: linear-gradient(100deg, #4da88e, #5878bd);
    color: white;
    font-weight: 700;
    min-height: 44px;
}


.stButton > button:hover {
    border-color: #a5ead5;
    color: #ffffff;
}

div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    background: rgba(24,27,38,0.55);
    overflow: hidden;
}


/* Input */

.stSelectbox label,
.stNumberInput label {
    color: #c8cbd6 !important;
    font-weight: 600;
}


/* Dataframe */

[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}


/* Info boxes */

.stAlert {
    border-radius: 14px;
}


@media (max-width: 760px) {
    .block-container {
        padding: 0.8rem 1rem 3rem;
    }

    .hero-title {
        font-size: 38px;
    }

    .navbar {
        margin-bottom: 17px;
    }

    div[role="radiogroup"] {
        justify-content: flex-start;
        flex-wrap: wrap;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("netflix_users.csv")

    data.columns = data.columns.str.strip()

    return data


try:

    df = load_data()

except Exception as e:

    st.error("Could not load netflix_users.csv")

    st.write(e)

    st.stop()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "User_ID",
    "Name",
    "Age",
    "Country",
    "Subscription_Type",
    "Watch_Time_Hours",
    "Favorite_Genre",
    "Last_Login"
]


missing = [
    c for c in required_columns
    if c not in df.columns
]


if missing:

    st.error(
        f"Missing columns: {missing}"
    )

    st.stop()


# ============================================================
# PREPROCESSING
# ============================================================

df["Age"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

df["Watch_Time_Hours"] = pd.to_numeric(
    df["Watch_Time_Hours"],
    errors="coerce"
)


df["Age"] = df["Age"].fillna(
    df["Age"].median()
)


df["Watch_Time_Hours"] = df[
    "Watch_Time_Hours"
].fillna(
    df["Watch_Time_Hours"].median()
)


df["Subscription_Type"] = df[
    "Subscription_Type"
].fillna(
    df["Subscription_Type"].mode()[0]
)


df["Favorite_Genre"] = df[
    "Favorite_Genre"
].fillna(
    df["Favorite_Genre"].mode()[0]
)


# ============================================================
# ENCODING
# ============================================================

subscription_encoder = LabelEncoder()

genre_encoder = LabelEncoder()


df["Subscription_Encoded"] = (
    subscription_encoder.fit_transform(
        df["Subscription_Type"].astype(str)
    )
)


df["Genre_Encoded"] = (
    genre_encoder.fit_transform(
        df["Favorite_Genre"].astype(str)
    )
)


features = [
    "Age",
    "Watch_Time_Hours",
    "Subscription_Encoded",
    "Genre_Encoded"
]


X = df[features]


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# TOP NAVIGATION
# ============================================================

def navigate_to(page_name):
    st.session_state["app_page"] = page_name


brand_col, nav_col = st.columns([1.2, 5.8])

with brand_col:
    st.markdown(
        '<div class="brand-lockup">'
        '<div class="brand-mark">◈</div>'
        '<div class="logo">Cluster<span>IQ</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

with nav_col:
    page = st.radio(
        "Navigation",
        [
            "Home",
            "Dataset",
            "Training",
            "Segments",
            "Prediction"
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="app_page"
    )

st.markdown('<div class="navbar"></div>', unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    hero_copy, hero_visual = st.columns([1.08, 0.92], gap="large")

    with hero_copy:
        st.markdown(
            '<div class="hero-badge">CUSTOMER INTELLIGENCE · CLUSTERING</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="hero-title">'
            'See the shape of your customers.<br>'
            '<span class="gradient-text">Find the segments.</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="hero-description">'
            'Explore how viewing habits, age, subscription plans, '
            'and genre preferences reveal distinct customer groups.'
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")
        action_col, secondary_col = st.columns([1.1, 1.2])
        with action_col:
            st.button(
                "Train a model  →",
                on_click=navigate_to,
                args=("Training",),
                use_container_width=True
            )
        with secondary_col:
            st.button(
                "Explore segments",
                on_click=navigate_to,
                args=("Segments",),
                use_container_width=True
            )

    with hero_visual:
        st.markdown(
            '<div class="visual-heading">'
            '<span>Customer behavior map</span>'
            '<span class="visual-status">● DATA OVERVIEW</span>'
            '</div>'
            '<div class="hero-note">Age and viewing time, colored by plan</div>',
            unsafe_allow_html=True
        )
        sample = df.sample(min(2500, len(df)), random_state=42)
        fig = px.scatter(
            sample,
            x="Age",
            y="Watch_Time_Hours",
            color="Subscription_Type",
            hover_data=["Favorite_Genre"],
            color_discrete_sequence=["#78d6b8", "#9caaff", "#f2a477", "#e991bd"]
        )
        fig.update_traces(marker={"size": 7, "opacity": 0.78})
        fig.update_layout(
            height=330,
            margin={"l": 8, "r": 8, "t": 12, "b": 8},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#b8becc", "family": "DM Sans"},
            legend={"orientation": "h", "y": 1.12, "x": 0},
            xaxis={"title": "Age", "gridcolor": "rgba(255,255,255,0.06)"},
            yaxis={"title": "Watch time (hours)", "gridcolor": "rgba(255,255,255,0.06)"}
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )
        st.markdown(
            '<div class="hero-tags">'
            '<span class="hero-tag">K-Means</span>'
            '<span class="hero-tag">Silhouette score</span>'
            '<span class="hero-tag">4 customer signals</span>'
            '<span class="hero-tag">Cluster profiles</span>'
            '</div>',
            unsafe_allow_html=True
        )


    st.write("")
    st.write("")


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    metrics = [
        (
            "👥",
            "Customers",
            f"{len(df):,}"
        ),

        (
            "🎂",
            "Average Age",
            f"{df['Age'].mean():.1f} yrs"
        ),

        (
            "⏱️",
            "Avg Watch Time",
            f"{df['Watch_Time_Hours'].mean():.2f} hrs"
        ),

        (
            "🎯",
            "ML Features",
            "4"
        )
    ]


    for column, item in zip(
        [col1, col2, col3, col4],
        metrics
    ):

        with column:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                {item[0]} {item[1]}
                </div>

                <div class="metric-value">
                {item[2]}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.markdown(
        '<div class="section-title">'
        '📄 Customer Dataset'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-description">'
        'Explore the Netflix customer data used to train the model.'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Rows",
        f"{len(df):,}"
    )


    c2.metric(
        "Columns",
        len(df.columns)
    )


    c3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=450
    )


    st.subheader(
        "Features used for clustering"
    )


    feature_info = pd.DataFrame({

        "Feature": [
            "Age",
            "Watch_Time_Hours",
            "Subscription_Type",
            "Favorite_Genre"
        ],

        "Description": [
            "Customer age",
            "Total viewing hours",
            "Customer subscription plan",
            "Preferred content genre"
        ]

    })


    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TRAINING
# ============================================================

elif page == "Training":

    st.markdown(
        '<div class="section-title">'
        '🤖 Model Training'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-description">'
        'Train K-Means and discover natural customer groups.'
        '</div>',
        unsafe_allow_html=True
    )


    k = st.slider(
        "Number of Clusters (K)",
        2,
        10,
        4
    )


    # --------------------------------------------------------
    # ELBOW
    # --------------------------------------------------------

    st.subheader(
        "📈 Elbow Method"
    )


    wcss = []


    for i in range(2, 11):

        temp = KMeans(
            n_clusters=i,
            random_state=42,
            n_init=10
        )

        temp.fit(X_scaled)

        wcss.append(
            temp.inertia_
        )


    elbow_df = pd.DataFrame({

        "K": range(2, 11),

        "WCSS": wcss

    })


    fig = px.line(
        elbow_df,
        x="K",
        y="WCSS",
        markers=True
    )


    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if st.button(
        "🚀 Train K-Means Model",
        type="primary"
    ):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )


        labels = model.fit_predict(
            X_scaled
        )


        score = silhouette_score(
            X_scaled,
            labels
        )


        model_data = {

            "model": model,

            "scaler": scaler,

            "subscription_encoder":
                subscription_encoder,

            "genre_encoder":
                genre_encoder,

            "features":
                features,

            "k":
                k

        }


        with open(
            "trained_model.pkl",
            "wb"
        ) as f:

            pickle.dump(
                model_data,
                f
            )


        st.success(
            "✓ Model trained successfully"
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Clusters",
            k
        )


        c2.metric(
            "Silhouette Score",
            f"{score:.3f}"
        )


        c3.metric(
            "Customers",
            f"{len(df):,}"
        )


        # ----------------------------------------------------
        # CLUSTER DATA
        # ----------------------------------------------------

        result = df.copy()

        result["Cluster"] = labels


        st.subheader(
            "Customer Distribution"
        )


        counts = (
            result["Cluster"]
            .value_counts()
            .sort_index()
        )


        fig2 = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={
                "x": "Cluster",
                "y": "Customers"
            }
        )


        st.plotly_chart(
            fig2,
            use_container_width=True
        )


        st.subheader(
            "Cluster Visualization"
        )


        plot_data = result.sample(
            min(5000, len(result)),
            random_state=42
        )


        plot_data["Cluster"] = (
            plot_data["Cluster"].astype(str)
        )


        fig3 = px.scatter(
            plot_data,
            x="Age",
            y="Watch_Time_Hours",
            color="Cluster",
            hover_data=[
                "Subscription_Type",
                "Favorite_Genre"
            ]
        )


        fig3.update_layout(
            height=550
        )


        st.plotly_chart(
            fig3,
            use_container_width=True
        )


# ============================================================
# SEGMENTS
# ============================================================

elif page == "Segments":

    st.markdown(
        '<div class="section-title">'
        '👥 Customer Segments'
        '</div>',
        unsafe_allow_html=True
    )


    if not os.path.exists(
        "trained_model.pkl"
    ):

        st.warning(
            "Train the K-Means model first."
        )

        st.stop()


    with open(
        "trained_model.pkl",
        "rb"
    ) as f:

        saved = pickle.load(f)


    model = saved["model"]


    labels = model.predict(
        X_scaled
    )


    result = df.copy()

    result["Cluster"] = labels


    st.subheader(
        "Cluster Overview"
    )


    distribution_col, behavior_col = st.columns(2)


    with distribution_col:

        cluster_counts = (
            result["Cluster"]
            .value_counts()
            .sort_index()
        )

        distribution_fig = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            labels={"x": "Customer cluster", "y": "Customers"},
            color=cluster_counts.index.astype(str),
            color_discrete_sequence=[
                "#78d6b8", "#9caaff", "#f2a477", "#e991bd",
                "#e5cc72", "#70b4d8", "#d78ac6", "#9bc178",
                "#df8d83", "#9da4b2"
            ]
        )
        distribution_fig.update_layout(
            showlegend=False,
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd0dd",
            xaxis={"title": "Customer cluster"},
            yaxis={"title": "Customers", "gridcolor": "rgba(255,255,255,0.06)"}
        )
        st.plotly_chart(distribution_fig, use_container_width=True)


    with behavior_col:

        segment_plot = result.sample(
            min(5000, len(result)),
            random_state=42
        ).copy()
        segment_plot["Cluster"] = segment_plot["Cluster"].astype(str)
        behavior_fig = px.scatter(
            segment_plot,
            x="Age",
            y="Watch_Time_Hours",
            color="Cluster",
            hover_data=["Subscription_Type", "Favorite_Genre"],
            color_discrete_sequence=[
                "#78d6b8", "#9caaff", "#f2a477", "#e991bd",
                "#e5cc72", "#70b4d8", "#d78ac6", "#9bc178",
                "#df8d83", "#9da4b2"
            ]
        )
        behavior_fig.update_traces(marker={"size": 6, "opacity": 0.72})
        behavior_fig.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd0dd",
            xaxis={"title": "Age", "gridcolor": "rgba(255,255,255,0.06)"},
            yaxis={"title": "Watch time (hours)", "gridcolor": "rgba(255,255,255,0.06)"}
        )
        st.plotly_chart(behavior_fig, use_container_width=True)


    summary = result.groupby(
        "Cluster"
    ).agg(

        Customers=("User_ID", "count"),

        Average_Age=("Age", "mean"),

        Average_Watch_Time=(
            "Watch_Time_Hours",
            "mean"
        )

    ).round(2)


    st.dataframe(
        summary,
        use_container_width=True
    )


    st.subheader(
        "Segment Profiles"
    )


    overall_watch = (
        df["Watch_Time_Hours"].mean()
    )


    for cluster in summary.index:

        cluster_data = result[
            result["Cluster"] == cluster
        ]


        avg_watch = (
            cluster_data[
                "Watch_Time_Hours"
            ].mean()
        )


        avg_age = (
            cluster_data["Age"].mean()
        )


        users = len(cluster_data)


        common_genre = (
            cluster_data[
                "Favorite_Genre"
            ].mode()
        )


        genre = (
            common_genre.iloc[0]
            if len(common_genre)
            else "N/A"
        )


        common_subscription = (
            cluster_data[
                "Subscription_Type"
            ].mode()
        )


        subscription = (
            common_subscription.iloc[0]
            if len(common_subscription)
            else "N/A"
        )


        if avg_watch > overall_watch * 1.2:

            segment = "🔥 High Engagement"

        elif avg_watch < overall_watch * 0.8:

            segment = "🌱 Low Engagement"

        else:

            segment = "⭐ Moderate Engagement"


        st.markdown(
            f"""
            <div class="metric-card">

            <h3>
            Cluster {cluster} — {segment}
            </h3>

            <p>
            <b>{users:,}</b> customers
            &nbsp;&nbsp; | &nbsp;&nbsp;
            Average Age: <b>{avg_age:.1f}</b>
            &nbsp;&nbsp; | &nbsp;&nbsp;
            Average Watch Time:
            <b>{avg_watch:.1f} hrs</b>
            </p>

            <p>
            Common Subscription:
            <b>{subscription}</b>
            &nbsp;&nbsp; | &nbsp;&nbsp;
            Favorite Genre:
            <b>{genre}</b>
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.markdown(
        '<div class="section-title">'
        '🔮 Predict Customer Segment'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-description">'
        'Enter a new customer profile and use the trained '
        'K-Means model to identify the closest customer segment.'
        '</div>',
        unsafe_allow_html=True
    )


    if not os.path.exists(
        "trained_model.pkl"
    ):

        st.warning(
            "Please train the model first."
        )

        st.stop()


    with open(
        "trained_model.pkl",
        "rb"
    ) as f:

        saved = pickle.load(f)


    model = saved["model"]

    saved_scaler = saved["scaler"]

    sub_encoder = (
        saved["subscription_encoder"]
    )

    genre_encoder_saved = (
        saved["genre_encoder"]
    )


    c1, c2 = st.columns(2)


    with c1:

        age = st.number_input(
            "Age",
            10,
            100,
            25
        )


        watch_time = st.number_input(
            "Watch Time Hours",
            0.0,
            5000.0,
            500.0,
            step=10.0
        )


    with c2:

        subscription = st.selectbox(
            "Subscription Type",
            sorted(
                df[
                    "Subscription_Type"
                ].unique()
            )
        )


        genre = st.selectbox(
            "Favorite Genre",
            sorted(
                df[
                    "Favorite_Genre"
                ].unique()
            )
        )


    st.write("")


    if st.button(
        "🔮 Analyze Customer",
        type="primary"
    ):

        sub_encoded = (
            sub_encoder.transform(
                [subscription]
            )[0]
        )


        genre_encoded = (
            genre_encoder_saved.transform(
                [genre]
            )[0]
        )


        new_customer = np.array([
            [
                age,
                watch_time,
                sub_encoded,
                genre_encoded
            ]
        ])

        new_customer = pd.DataFrame(
            new_customer,
            columns=saved["features"]
        )

        scaled_customer = (
            saved_scaler.transform(
                new_customer
            )
        )

        cluster = model.predict(
            scaled_customer
        )[0]


        all_labels = model.predict(
            X_scaled
        )


        analysis = df.copy()

        analysis["Cluster"] = all_labels


        cluster_data = analysis[
            analysis["Cluster"] == cluster
        ]


        avg_age = (
            cluster_data["Age"].mean()
        )


        avg_watch = (
            cluster_data[
                "Watch_Time_Hours"
            ].mean()
        )


        users = len(cluster_data)


        overall = (
            df["Watch_Time_Hours"].mean()
        )


        if avg_watch > overall * 1.2:

            segment = "🔥 High Engagement"

        elif avg_watch < overall * 0.8:

            segment = "🌱 Low Engagement"

        else:

            segment = "⭐ Moderate Engagement"


        st.success(
            "✓ Customer analysis completed"
        )


        st.divider()


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Predicted Cluster",
            f"Cluster {cluster}"
        )


        c2.metric(
            "Customer Segment",
            segment
        )


        c3.metric(
            "Similar Customers",
            f"{users:,}"
        )


        st.subheader(
            "Prediction in the trained customer map"
        )


        prediction_plot = analysis.sample(
            min(5000, len(analysis)),
            random_state=42
        ).copy()
        prediction_plot["Cluster"] = prediction_plot["Cluster"].astype(str)
        prediction_fig = px.scatter(
            prediction_plot,
            x="Age",
            y="Watch_Time_Hours",
            color="Cluster",
            hover_data=["Subscription_Type", "Favorite_Genre"],
            color_discrete_sequence=[
                "#78d6b8", "#9caaff", "#f2a477", "#e991bd",
                "#e5cc72", "#70b4d8", "#d78ac6", "#9bc178",
                "#df8d83", "#9da4b2"
            ]
        )
        prediction_fig.add_trace(
            go.Scatter(
                x=[age],
                y=[watch_time],
                mode="markers",
                name="New customer",
                marker={
                    "symbol": "star",
                    "size": 19,
                    "color": "#ffffff",
                    "line": {"color": "#10131b", "width": 2}
                },
                hovertemplate=(
                    "New customer<br>Age: %{x}<br>Watch time: %{y} hrs"
                    "<extra></extra>"
                )
            )
        )
        prediction_fig.update_layout(
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd0dd",
            xaxis={"title": "Age", "gridcolor": "rgba(255,255,255,0.06)"},
            yaxis={"title": "Watch time (hours)", "gridcolor": "rgba(255,255,255,0.06)"}
        )
        st.plotly_chart(prediction_fig, use_container_width=True)


        st.subheader(
            "Customer Profile"
        )


        profile = pd.DataFrame({

            "Attribute": [
                "Age",
                "Watch Time",
                "Subscription",
                "Favorite Genre"
            ],

            "Value": [
                f"{age} years",
                f"{watch_time:.1f} hours",
                subscription,
                genre
            ]

        })

        st.table(profile)


        st.subheader(
            "Similar Customer Group"
        )


        c1, c2 = st.columns(2)


        c1.metric(
            "Cluster Average Age",
            f"{avg_age:.1f} years"
        )


        c2.metric(
            "Cluster Average Watch Time",
            f"{avg_watch:.1f} hrs"
        )


        st.info(
            f"This customer has been assigned to "
            f"**Cluster {cluster}** because their "
            f"profile is similar to customers in this "
            f"cluster based on the trained K-Means model."
        )
