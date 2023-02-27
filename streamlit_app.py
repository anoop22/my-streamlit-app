import streamlit as st

# Display a text input widget for the user's name
name = st.text_input("Enter your name")

# If the user has entered a name, display a personalized greeting
if name:
    st.write(f"Hello, {name}!")
