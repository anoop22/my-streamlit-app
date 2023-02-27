# -*- coding: utf-8 -*-
"""StreamLitUML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qnzvpfsn5BXDqrHNdSzxv4s0tfe3MeIv
"""

import streamlit as st
import openai
import plantuml

# initialize the OpenAI API with your API key
openai.api_key = st.secrets["api_secret"]

# specify the URL for the PlantUML server
url = "http://www.plantuml.com/plantuml/img/"

# prompt the user for input
user_input = st.text_input("Enter the text for which you want to generate a diagram:")
type_input = st.text_input("Enter which kind of diagram (one word only): ")

if st.button("Generate diagram"):
    # use the OpenAI API to generate PlantUML syntax
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"I just output the PlantUML code based on input text and nothing else. I generate the plantuml code using plantuml's existing syntax. PlantUML code for generating {type_input} based on the following text:\n\n Start of Text \n\n{user_input} \n\n End of Text",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # extract the generated PlantUML syntax from the API response
    generated_syntax = response.choices[0].text.strip()
    
    #st.text_input("generated_syntax", generated_syntax)
    
    # generate the diagram using a different variable name than Image 
    diagram_bytes = plantuml.PlantUML(url=url).processes(generated_syntax)
    my_diagram = diagram_bytes
    
    # display the diagram using Image class from IPython.display 
    from IPython.display import Image 
    Image(my_diagram)

    # display the diagram using Streamlit's Image component
    st.image(my_diagram)
    
    st.code(generated_syntax)