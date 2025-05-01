import streamlit as st
from openai import OpenAI
import pyttsx3

import json
import speech_recognition as sr
from io import BytesIO

from sentence_transformers import SentenceTransformer, util


st.set_page_config(page_title="Duolingo German", page_icon="🇩🇪", layout="wide")


if "mic_test" not in st.session_state:
    st.session_state.mic_test = False





@st.dialog("Lets test your microphone and speaker before we proceed further")
def dialog_box():

    st.subheader("Speaker Test")
    st.write("Please listen to the audio below to check if your speaker is working.")
    engine = pyttsx3.init(driverName='sapi5')
    engine.save_to_file("Hello, this is a test of your microphone and speaker. Please make sure they are working properly.", "test.mp3")
    engine.runAndWait()
    st.audio("test.mp3", format="audio/mp3")
    st.subheader("Microphone Test")
    st.write("Please say the phrase in the audio player above and press the button below to check if your microphone is working.")
    st.write("Phrase: Hello, this is a test of your microphone and speaker. Please make sure they are working properly. ")
    mic_audio = st.audio_input("**Press the button below to record your voice**")
    if mic_audio is not None:
        st.success("**Your microphone is working properly to proceed with the questions**")

        # st.button("**Proceed to the next question**")
        st.session_state.mic_test = True
        st.rerun()
def jaccard_similarity(a, b):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')  # Good for German
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)

    return similarity


def transcribe_audio(audio_file):
    audio_bytes = audio_file.read()
    audio_buffer = BytesIO(audio_bytes)
    audio_buffer.seek(0)
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_buffer) as source:
        audio_data = recognizer.record(source)
    transcription = recognizer.recognize_google(audio_data, language='de-DE')
    return transcription



def play_audio(text):
    engine = pyttsx3.init(driverName='sapi5')
    engine.save_to_file(text, "test.mp3")
    engine.runAndWait()
    st.audio("test.mp3", format="audio/mp3")


@st.fragment
def question(response):
    # Initialize 'x' in session_state if it doesn't exist
    st.subheader("**Question-**")
    response = response['data']
    length = len(response)
    if 'x' not in st.session_state:
        st.session_state.x = 0
    german_phrase = list(response.keys())
    english_phrase = list(response.values())
    try:
        st.write(f":violet[German Phrase:]  " + german_phrase[st.session_state.x])
        st.write(f":violet[English Phrase:] " + english_phrase[st.session_state.x])
        play_audio(list(response.keys())[st.session_state.x])  


    except:
        st.session_state.x = 0
        st.write("**You have completed all the questions!**")
        st.balloons()
        return  True
    
    user_audio_input = st.audio_input("**Answer below**", key = "mic_input")

    if st.session_state.mic_input is not None :
        transcribed_text = transcribe_audio(user_audio_input)
        with st.spinner("**Checking your answer...**"):
            jaccard = jaccard_similarity(german_phrase[st.session_state.x], transcribed_text)

        
            if jaccard.item() > 0.1:
                st.success("**Correct!**")
                st.write(f":green[**Your Accuracy is : {round(jaccard.item()*100,2)}%**]")
                submit = st.button("**Proceed to the next question**")
                user_audio_input = None
                if submit:
                    del st.session_state.mic_input
                    st.session_state.x += 1
                    st.rerun(scope='fragment')
                    
            else:
                st.error("**Incorrect! Please try again.**")
                st.write(f"**Your answer: {transcribed_text}**")
                st.write(f"**Correct answer: {german_phrase[st.session_state.x]}**")
                st.write(f":red[**Your Accuracy is : {round(jaccard.item()*100,2)}%**]")
                del st.session_state.mic_input

def generate_phrases(level):
    

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {
        "role": "system",
        "content": (
            "You are a helpful assistant that helps the user to learn German language. "
            "You will give the user a phrase in German which can be used in a conversation. "
            "You will also give the user the phrase in English. You will give the user 5 phrases "
            f"to practise for {level} level."
        )
        },
        {
        "role": "user",
        "content": (
            f"Give 3 phrases in German for {level} level in form of a dictionary with key as the phrase "
            "in German and value as the phrase in English. The phrases should be in a dictionary format "
            "and give only the dictionary. Do not give any other text."
        )
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "phrase_schema",
        "description": "A dictionary of German phrases and their English translations.",
        "schema": {
            "type": "object",
            "properties": {
            "data": {  # You can explicitly define 'data' here if you want, though not strictly required
                "type": "object",
                "additionalProperties": {
                "type": "string",
                "description": "The English translation of the German phrase."
                }
            }
            },
            "required": ["data"]  # Ensure the 'data' field is marked as required
        }
        }
    }
    )
    return completion

    
col1, col2 , col3 = st.columns(3)
col2.title(":blue[German Language Learning]")
st.subheader(f"**Welcome to your personalized German learning assistant! Choose your difficulty level — Easy, Medium, or Hard — and get 5 useful German phrases tailored to your skill level. Whether you're just starting out or brushing up on your skills, this app is here to make learning simple, fast, and fun.**")
col1, col2 , col3 = st.columns(3)
col2.subheader(":violet[**Select your level**]")
level = col2.segmented_control(" ", options= ["**Easy**", "**Medium**", "**Hard**"], selection_mode= "single")

if level is not None and st.session_state.mic_test == False:
    # st.write(f"**You have selected the level: {level}**")
    dialog_box()
    
if st.session_state.mic_test:     

    # question = st.button("**Proceed to the next question**")
    st.session_state.mic_test = False
    completion = generate_phrases(level)
    st.subheader(f" {level} level is selected")
    st.divider()
    response = completion.choices[0].message.content
    response = json.loads(response)
    with st.container():
        #
        f =question(response)
        # st.write(finished) 



