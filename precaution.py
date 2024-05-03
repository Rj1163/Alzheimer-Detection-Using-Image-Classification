import streamlit as st
import wikipediaapi


def precautions_page():
    st.subheader('Precautions for Alzheimer\'s Disease')

    # Create a Wikipedia object with a specified user agent
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='streamlit-alzheimers-app/1.0'
    )

    page = wiki_wiki.page("Prevention of Alzheimer's disease")

    if page.exists():
        st.image("D:\\Projects\\Alzhiemer\\feat-crop-iStock-1026002288.jpg",
                 caption='Image of Alzheimer\'s Disease',
                 use_column_width=True)
        st.write(page.summary)
    else:
        st.write("Sorry, couldn't find information about the prevention of Alzheimer's disease on Wikipedia.")
