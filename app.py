import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Calories Predictor", page_icon="🍳", layout="centered")

st.title("🍳 AI Calories Predictor")
st.write("Paste your recipe, and the model will estimate its calorie content.")

# 3. Dropdown menu to select the model
available_models = [
    "TF-IDF + Ridge", 
    "Word2Vec + Random Forest", 
]
selected_model = st.selectbox("Select analytical model:", available_models)

recipe_text = st.text_area(
    "Recipe text:", 
    height=200, 
    placeholder="E.g. Mix 200g of butter with two cups of flour and add 3 eggs..."
)

if st.button("Calculate Calories", type="primary"):
    if recipe_text.strip() == "":
        st.warning("Please enter a recipe first!")
    else:
        with st.spinner('The model is analyzing the text...'):
            
            time.sleep(1.5) # Simulating thinking
            
            calculated_calories = 540 
            
            st.success(f"Analysis completed! Model used: {selected_model}")
            
            output_data = {
                "Metric": ["Calories (kcal)"],
                "Estimated Value": [f"{calculated_calories} kcal"]
            }
            df = pd.DataFrame(output_data)
            
            st.table(df.set_index("Metric"))