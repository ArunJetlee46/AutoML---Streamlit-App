import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

def train_model(df):
    st.write("##  Machine Learning Model Training")

    if df is None or df.empty:
        st.warning("⚠️ Please upload a valid dataset before training.")
        return

    # Sidebar for model selection
    st.sidebar.header("⚙️ Model Training Options")

    # Select Target Column
    target = st.sidebar.selectbox("🎯 Select Target Variable", df.columns, index=len(df.columns)-1)

    # Encode categorical target variable if needed
    if df[target].dtype == 'object':
        st.sidebar.write("🔠 Encoding categorical target variable...")
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])

    # Feature & Target Splitting
    X = df.drop(columns=[target])
    y = df[target]

    # Ensure the dataset isn't empty
    if X.empty:
        st.error("❌ No valid features available for training. Please check preprocessing steps.")
        return

    # Determine Task: Classification or Regression
    if len(y.unique()) > 10:  # If more than 10 unique values, assume regression
        task_type = "Regression"
    else:
        task_type = "Classification"

    st.sidebar.write(f"🔍 Detected Task Type: **{task_type}**")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Suggested Models
    if task_type == "Classification":
        suggested_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
            "Random Forest": RandomForestClassifier(class_weight="balanced"),
            "SVM": SVC(class_weight="balanced"),
            "Naive Bayes": GaussianNB()
        }
    else:
        suggested_models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "SVM": SVR()
        }

    # Model Selection
    model_choice = st.sidebar.selectbox("🤖 Select Machine Learning Model", list(suggested_models.keys()))

    # Hyperparameter tuning (for certain models)
    if model_choice in ["Random Forest", "Decision Tree"]:
        max_depth = st.sidebar.slider("🌳 Max Depth", 2, 20, 10)
    if model_choice == "Random Forest":
        n_estimators = st.sidebar.slider("🌲 Number of Trees", 10, 300, 100)
    if model_choice == "SVM":
        kernel = st.sidebar.selectbox("🔧 Kernel Type", ["linear", "rbf", "poly"])

    # Initialize Model with Selected Parameters
    try:
        if model_choice == "Random Forest":
            if task_type == "Classification":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight="balanced")
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        elif model_choice == "Decision Tree":
            if task_type == "Classification":
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight="balanced")
            else:
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

        elif model_choice == "SVM":
            if task_type == "Classification":
                model = SVC(kernel=kernel, class_weight="balanced")
            else:
                model = SVR(kernel=kernel)

        else:
            model = suggested_models[model_choice]

        # Train Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model Evaluation
        if task_type == "Classification":
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model Accuracy: {accuracy:.2f}")

            # Show Classification Report
            st.subheader("📊 Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        else:  # Regression Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.success(f"✅ Mean Squared Error (MSE): {mse:.2f}")
            st.success(f"✅ R-squared Score: {r2:.2f}")

            # Scatter Plot of Predictions vs Actual
            st.subheader("📊 Predicted vs. Actual Values")
            results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            fig = px.scatter(results_df, x="Actual", y="Predicted", trendline="ols")
            st.plotly_chart(fig)

        # Feature Importance (for tree-based models)
        if model_choice in ["Random Forest", "Decision Tree"]:
            st.subheader("🔍 Feature Importance")
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            feature_importances.plot(kind="bar", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ An error occurred while training the model: {str(e)}")
