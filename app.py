import streamlit as st 
import joblib 

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("News_detection_model.jb")

st.title("Fake news detector")
st.write("Enter a piece of news article to detect whether its fake or real")

input = st.text_area("News Article","")

if st.button("Detect Now"):
    if input.strip():
        transform_input = vectorizer.transform([input])
        prediction = model.predict(transform_input)
        
        if prediction[0] == 0:
            st.success("The news is accurate!")
            
        else:
            st.error("The news is fake!")
            
    else:
        st.warning("You must enter news for detection")