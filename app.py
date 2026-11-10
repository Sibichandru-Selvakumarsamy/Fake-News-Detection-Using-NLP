

import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Fake News Detection Using NLP",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection Using NLP")
st.metric("Model Test Accuracy", "99.01%")
st.divider()

st.write(
    "Analyze a news article using Machine Learning "
    "and Natural Language Processing."
)

st.info(
    "Enter the news article below and click Detect News."
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")


st.write("Model path:", model_path)
st.write("Model file exists:", os.path.exists(model_path))


if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    def clear_text():
        st.session_state["news_input"] = ""

    news = st.text_area(
        "📝 Enter your news article:",
        height=200,
        placeholder="Paste the complete news article here...",
        key="news_input"
    )

    st.button("🗑️ Clear Text", on_click=clear_text)

    if st.button("🔍 Detect News", use_container_width=True):
        if news.strip():
            prediction = model.predict([news])[0]
            probabilities = model.predict_proba([news])[0]
            confidence = max(probabilities) * 100

            if str( prediction).upper() == "FAKE":
                st.error("🚨 Prediction: Fake News")
            elif str(prediction).upper() == "REAL":
                st.error("Prediction: Real News")    
            else:
                st.success("✅ Prediction: Real News")

            st.write(f"Model confidence: {confidence:.2f}%")
        else:
            st.warning("Please enter a news article first.")

else:
    st.error("Model not found. Please run train_model.py first.")


st.divider()

st.subheader("📌 About This Project")

st.write("""
This project detects fake and real news articles using
Natural Language Processing (NLP) and Machine Learning.

**Technologies Used:**
- Python
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

**Dataset:** Fake.csv and True.csv
""")


st.divider()

st.markdown(
    """
    <div style="text-align: center;">
        <p><b>Fake News Detection Using NLP</b></p>
        <p>Developed using Python, Machine Learning and Streamlit</p>
        <p>Academic Project | Artificial Intelligence & Data Science</p>
    </div>
    """,
    unsafe_allow_html=True
)
