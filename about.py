import streamlit as st
import wikipediaapi


def about_page():
    st.subheader('About Alzheimer\'s Disease')

    st.image("D:\\Projects\\Alzhiemer\\alzheimers_final_flat.jpg",
             caption='Image of Alzheimer\'s Disease',
             use_column_width=True)

    # Create a Wikipedia object with a specified user agent
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='streamlit-alzheimers-app/1.0'
    )

    page = wiki_wiki.page("Alzheimer's disease")

    if page.exists():

        st.write(page.summary)
    else:
        st.write("Sorry, couldn't find information about Alzheimer's disease on Wikipedia.")

    st.image("D:\\Projects\\Alzhiemer\\Alzheimers Brain.jpg",
             caption='Image of Alzheimer\'s Disease',
             use_column_width=True)

