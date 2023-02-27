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
#key_input = st.text_input("Enter your Open AI API Key:")
openai.api_key = "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"

# specify the URL for the PlantUML server
url = "http://www.plantuml.com/plantuml/img/"

# prompt the user for input
user_input = st.text_input("Enter the text for which you want to generate a diagram:")
type_input = st.text_input("Enter which kind of diagram (one word only): ")

if st.button("Generate diagram"):
    # use the OpenAI API to generate PlantUML syntax
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"I just output the PlantUML syntax based on input text and nothing else. If i can't generate a plantuml syntax then i generate a sample plantuml syntax. Following is the PlantUML syntax for a {type_input} based on the following text:\n\n{user_input}",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # extract the generated PlantUML syntax from the API response
    generated_syntax = response.choices[0].text.strip()

    # generate the diagram using a different variable name than Image 
    diagram_bytes = plantuml.PlantUML(url=url).processes(generated_syntax)
    my_diagram = diagram_bytes.decode('utf-8')

    # display the diagram using Streamlit's Image component
    st.image(my_diagram)