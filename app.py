import streamlit as st
from model import retrieve_company_info

st.title("Extended Company Information Retrieval")

# Input for company name
company_name = st.text_input("Enter the company name:")

if st.button("Retrieve Information"):
    if company_name:
        info = retrieve_company_info(company_name)
        if info:
            st.write("Company Information:")
            st.json(info)
        else:
            st.write("No information found for the specified company.")
    else:
        st.write("Please enter a company name.")
