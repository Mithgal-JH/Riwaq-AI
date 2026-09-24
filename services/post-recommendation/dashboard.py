import streamlit as st
import requests
import pandas as pd
import datetime

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="BinX Recommendation Engine", layout="wide", page_icon="🚀")

# Initialize session state for mock interactions
if "recent_interactions" not in st.session_state:
    st.session_state["recent_interactions"] = []

st.title("🚀 BinX AI: Post Recommendation Dashboard")
st.markdown("Interactive dashboard for the Week 9/10 AI Internship Capstone. Watch the recommendation engine update dynamically based on user events!")

# --- Sidebar Controls ---
st.sidebar.header("🔧 Controls")
user_id = st.sidebar.text_input("Simulate User ID", value="usr_new_123")
limit = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

if st.sidebar.button("Check API Health"):
    try:
        res = requests.get(f"{API_BASE_URL}/health")
        if res.status_code == 200:
            st.sidebar.success(f"API Online: {res.json()}")
        else:
            st.sidebar.error("API is offline or returning an error.")
    except Exception as e:
        st.sidebar.error(f"Failed to connect: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Simulate User Interaction")
st.sidebar.markdown("*Note: In a live environment, the .NET backend tracks these and passes them to our API.*")
simulate_post_id = st.sidebar.text_input("Post ID to interact with", value="pst_101")
interaction_type = st.sidebar.selectbox("Interaction Type", ["like", "save", "repost"])

if st.sidebar.button("Record Interaction"):
    st.session_state["recent_interactions"].append({
        "post_id": simulate_post_id,
        "interaction_type": interaction_type,
        "timestamp": datetime.datetime.now().isoformat() + "Z"
    })
    st.sidebar.success(f"Interaction '{interaction_type}' on {simulate_post_id} recorded locally!")

# --- Main Content Area ---
st.header(f"Feed for `{user_id}`")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Personalized Timeline")
    if st.button("Generate Recommendations"):
        with st.spinner("Calculating semantic similarities & topic affinities..."):
            try:
                # Generate a list of all mock candidate IDs from pst_101 to pst_805
                all_candidates = [f"pst_{topic}0{idx}" for topic in range(1, 9) for idx in range(1, 6)]
                
                payload = {
                    "request_id": f"req_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "user_id": user_id,
                    "limit": limit,
                    "declared_topics": ["Programming/Web", "AI/Data"],
                    "learning_direction": "Backend Development",
                    "eligible_candidate_ids": all_candidates,
                    "recent_interactions": st.session_state["recent_interactions"]
                }
                res = requests.post(f"{API_BASE_URL}/api/v1/recommendations/posts", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    recommendations = data.get("items", [])
                    if not recommendations:
                        st.info("No recommendations found. Is the catalog synced?")
                    
                    for i, rec in enumerate(recommendations):
                        badge_html = "".join([f"<span style='background-color:#0078D4;color:white;padding:2px 6px;border-radius:4px;font-size:12px;margin-right:4px;'>{code}</span>" for code in rec['reason_codes']])
                        
                        st.markdown(f"**{i+1}. {rec['id']}** (Score: `{rec['score']:.4f}`) — {badge_html}", unsafe_allow_html=True)
                        st.progress(min(rec['score'], 1.0))
                else:
                    st.error(f"Error fetching recommendations: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to Recommendation API: {e}")

with col2:
    st.subheader("System Metrics")
    st.info("💡 **How it works:**")
    st.markdown("""
    - **Semantic Search:** Uses `all-MiniLM-L6-v2` for dense embeddings.
    - **Topic Affinity:** Tracks interaction profiles passed from the backend.
    - **Creator Affinity:** Boosts posts from interacted creators.
    - **Freshness:** Applies time-decay penalty to older posts.
    """)
    st.markdown("---")
    st.warning("⚠️ **Limitation (Large Scale):** The current similarity matrix computation is O(n²) in memory. In a future iteration, we plan to move to row-wise similarity calculation.")
