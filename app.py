import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="EduPro Student Segmentation",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Segmentation & Personalized Course Recommendation")
st.markdown("### EduPro Analytics Dashboard")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    profiles = pd.read_csv("learner_profiles.csv")
    courses = pd.read_csv("courses_clean.csv")
    popularity = pd.read_csv("course_cluster_popularity.csv")
    return profiles, courses, popularity

profiles, courses, popularity = load_data()

# Load Models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("le_gender.pkl")
le_category = joblib.load("le_category.pkl")
le_level = joblib.load("le_level.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Learner Profile",
        "Recommendations",
        "Analytics",
        "About"
    ]
)

# -----------------------------
# Home Page
# -----------------------------
if page == "Home":

    st.subheader("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Learners", len(profiles))
    col2.metric("Courses", len(courses))
    col3.metric("Clusters", profiles["Cluster"].nunique())
    col4.metric("Categories", courses["CourseCategory"].nunique())

    st.divider()

    st.subheader("Cluster Distribution")

    cluster_count = (
        profiles["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_count.columns = ["Cluster", "Students"]

    fig = px.bar(
        cluster_count,
        x="Cluster",
        y="Students",
        text="Students",
        color="Cluster"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Learner Dataset")
    st.dataframe(profiles.head(20), use_container_width=True)
    # -----------------------------
# Learner Profile Page
# -----------------------------
elif page == "Learner Profile":

    st.subheader("👤 Learner Profile Explorer")

    user_ids = sorted(profiles["UserID"].unique())

    selected_user = st.selectbox(
        "Select User ID",
        user_ids
    )

    learner = profiles[profiles["UserID"] == selected_user].iloc[0]

    st.markdown("### Learner Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("*User ID:*", learner["UserID"])
        st.write("*Age:*", learner["Age"])
        st.write("*Gender:*", learner["Gender"])
        st.write("*Cluster:*", learner["Cluster"])

    with col2:
        st.write("*Preferred Category:*", learner["PreferredCategory"])
        st.write("*Total Courses:*", learner["TotalCoursesEnrolled"])
        st.write("*Average Rating:*", round(learner["AvgCourseRating"], 2))
        st.write("*Average Spending:*", round(learner["AvgSpending"], 2))

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            "Learning Depth Index",
            round(learner["LearningDepthIndex"], 2)
        )

    with col4:
        st.metric(
            "Diversity Score",
            round(learner["DiversityScore"], 2)
        )

    st.success(
        f"This learner belongs to Cluster {learner['Cluster']}."
    )
# -----------------------------
# Recommendation Page
# -----------------------------
elif page == "Recommendations":

    st.subheader("🎯 Personalized Course Recommendation")

    user_ids = sorted(profiles["UserID"].unique())

    selected_user = st.selectbox(
        "Select User ID",
        user_ids,
        key="recommend_user"
    )

    learner = profiles[profiles["UserID"] == selected_user].iloc[0]
    cluster = learner["Cluster"]

    st.info(f"Assigned Cluster: {cluster}")

    recommendations = popularity[
        popularity["Cluster"] == cluster
    ].sort_values(
        by="Score",
        ascending=False
    ).head(10)

    recommendations = recommendations.merge(
        courses,
        on="CourseID",
        how="left"
    )

    st.subheader("📚 Top Recommended Courses")

    st.dataframe(
        recommendations[
            [
                "CourseName",
                "CourseCategory",
                "CourseLevel",
                "CourseRating"
            ]
        ],
        use_container_width=True
    )

    fig = px.bar(
        recommendations,
        x="CourseName",
        y="CourseRating",
        color="CourseCategory",
        title="Recommended Course Ratings"
    )

    st.plotly_chart(fig, use_container_width=True)
    # -----------------------------
# Analytics Page
# -----------------------------
elif page == "Analytics":

    st.subheader("📊 Analytics Dashboard")

    col1, col2 = st.columns(2)

    # Course Category Distribution
    with col1:
        st.markdown("### Course Categories")

        category_count = (
            courses["CourseCategory"]
            .value_counts()
            .reset_index()
        )

        category_count.columns = ["Category", "Count"]

        fig = px.pie(
            category_count,
            names="Category",
            values="Count",
            title="Course Category Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Course Level Distribution
    with col2:
        st.markdown("### Course Levels")

        level_count = (
            courses["CourseLevel"]
            .value_counts()
            .reset_index()
        )

        level_count.columns = ["Level", "Count"]

        fig = px.bar(
            level_count,
            x="Level",
            y="Count",
            color="Level",
            text="Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### Cluster Distribution")

    cluster_count = (
        profiles["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_count.columns = ["Cluster", "Students"]

    fig = px.bar(
        cluster_count,
        x="Cluster",
        y="Students",
        color="Cluster",
        text="Students",
        title="Students in Each Cluster"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Preview")
    st.dataframe(courses.head(20), use_container_width=True)
    
    st.divider()
st.markdown("### 3D Learner Segmentation")

fig = px.scatter_3d(
    profiles,
    x="TotalCoursesEnrolled",
    y="AvgSpending",
    z="DiversityScore",
    color="Cluster",
    hover_name="UserID",
    title="Courses Enrolled vs Spending vs Diversity Score"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.markdown("### Average Spending by Cluster")

cluster_spending = profiles.groupby("Cluster")["AvgSpending"].mean().reset_index()

fig = px.bar(
    cluster_spending,
    x="Cluster",
    y="AvgSpending",
    color="Cluster",
    title="Average Spending by Cluster",
    text_auto=".2f"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.markdown("### Age Distribution")

fig = px.histogram(
    profiles,
    x="Age",
    nbins=10,
    title="Age Distribution of Learners"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.markdown("### Gender Distribution")

gender_count = profiles["Gender"].value_counts().reset_index()
gender_count.columns = ["Gender", "Count"]

fig = px.pie(
    gender_count,
    names="Gender",
    values="Count",
    title="Gender Distribution of Learners"
)

st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
# About Page
# -----------------------------
elif page == "About":

    st.subheader("ℹ️ About This Project")

    st.markdown("""
### Student Segmentation and Personalized Course Recommendation System for EduPro

This project uses *Machine Learning (K-Means Clustering)* to group learners based on their learning behavior.

### Features
- Learner Segmentation
- Personalized Course Recommendation
- Dashboard Analytics
- Interactive Visualizations
- Cluster-Based Insights

### Technologies Used
- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- Joblib

Developed as a B.Tech AIML academic project.
""")
