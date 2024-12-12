#Libraries
import streamlit as st
from transformers import pipeline
import sounddevice as sd
import numpy as np
import wave
import mysql.connector
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
import IPython.display as ipd
import ffmpeg
import re
from datetime import datetime
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

#global vars
schedule_data=None

# Whisper ASR Model
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Initialize the tokenizer from Hugging Face Transformers library
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql')
model = model.to(device)
model.eval()

def generate_sql(input_prompt):
    # Tokenize the input prompt
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    # Decode the output IDs to a string (SQL query in this case)
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_sql

def select_schedule_number(translation):
    word_to_id = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10
    }
    translation = translation.lower()

    for word, num_id in word_to_id.items():
        if word in translation:
            return num_id

    return None
# List of cities in Pakistan
PAKISTANI_CITIES = [
    "lahore", "islamabad", "karachi", "peshawar", "quetta", "multan", "faisalabad", 
    "rawalpindi", "hyderabad", "gujranwala", "sialkot", "bahawalpur", "sukkur"
]

def extract_trip_details(input_text):
    # Convert text to lowercase for easier matching
    text = input_text.lower()

    # Extract date in the format '12 december' or '12 dec'
    date_match = re.search(r'\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b', text)
    if not date_match:
        return None, None, None  # If date not found

    day = int(date_match.group(1))
    month_str = date_match.group(2)
    # Convert month to number
    month = datetime.strptime(month_str, "%B").month

    # Build date as 2024-MM-DD
    trip_date = f"2024-{month:02d}-{day:02d}"

    # Extract cities
    found_cities = [city for city in PAKISTANI_CITIES if city in text]
    if len(found_cities) < 2:
        return None, None, trip_date  # Return available info if not enough cities found

    # Assign cities
    start_city = found_cities[0]
    end_city = found_cities[1]

    query = f"""
        SELECT start_city, end_city, fare, departure_date, departure_time, capacity
        FROM schedule
        INNER JOIN bus ON bus.id = schedule.bus_id
        WHERE start_city = '{start_city}' AND end_city = '{end_city}' AND departure_date = '{trip_date}';
    """
    return query




def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host="localhost",      
            user="root",           
            password="kinghamzafami1a",  
            database="bus_reservation_system"
        )
        print("Database connection successful!")
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Function to fetch schedule data based on user source and destination location
def fetch_schedule_data(connection,query):
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        schedule_data = cursor.fetchall()
        cursor.close()
        return schedule_data
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
    
def fetch_fares_data(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT start_city, end_city, fare, departure_date, departure_time,capacity FROM schedule,bus where bus.id = schedule.bus_id;")
        schedule_data = cursor.fetchall()
        cursor.close()
        return schedule_data
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
def select_schedule_number(translation):
    word_to_id = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10
    }
    translation = translation.lower()

    for word, num_id in word_to_id.items():
        if word in translation:
            return num_id

    return None

def calculate_payment(fare,nos):
    return fare*nos
# Set the page configuration
st.set_page_config(page_title="Bus Reservation System", layout="centered")

# Define styles for improved visuals
def set_page_style():
    st.markdown(
        """
        <style>
            .main {
                background-color: #f7f9fc;
                text-align: center;
            }
            h1 {
                color: #2c3e50;
                font-size: 3em;
                margin-bottom: 10px;
            }
            .stButton>button {
                width: 100%;
                border-radius: 8px;
                font-size: 1.2em;
                color: white;
                background-color: #0073e6;
                border: none;
                padding: 10px;
                margin: 5px;
            }
            .stButton>button:hover {
                background-color: #005bb5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Main screen
def main():
    st.title("🚌 Bus Reservation System")
    st.markdown("### Welcome to the Bus Reservation System. Please choose an option below:")
    
    # Create buttons for Login and Signup
    col1, col2 = st.columns(2, gap="large")
    with col1:
        if st.button("Login"):
            st.experimental_set_query_params(page="login")

    with col2:
        if st.button("Signup"):
            st.experimental_set_query_params(page="signup")
def select_seats():
    st.title("🪑 Select Seats")
    params = st.experimental_get_query_params()
    fare = float(params.get("fare", [1])[0])  # Retrieve the number of seats from query paramet
    st.markdown("Please select the number of seats you want to reserve.")

    # Seat selection input
    num_seats = st.number_input("Number of Seats", min_value=1, max_value=10, step=1, value=1)

    if st.button("Proceed to Payment"):
        # Save the number of seats as a query parameter and move to the payment page
        st.experimental_set_query_params(page="payment", fare=fare*num_seats)
def payment():
    st.title("💳 Payment")
    params = st.experimental_get_query_params()
    fare = float(params.get("fare", [1])[0])  # Retrieve the number of seats from query parameters
    total_cost = fare

    # Display payment details
    st.markdown(f"### Total Cost: PKR {total_cost}")

    # Confirmation button
    if st.button("Confirm Payment"):
        st.success("Payment Successful! Your reservation has been recorded.")

# Login page
def login():
    st.title("🔑 Login")
    st.markdown("Please enter your login credentials below:")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Submit"):
    # Add authentication logic here
        if username and password:  # Replace with actual authentication logic
            st.success(f"Welcome back, {username}!")
            st.experimental_set_query_params(page="reservation")
        else:
            st.error("Invalid credentials. Please try again.")

# Function to record audio for 5 seconds
def record_voice():
    st.title("🎙️ Record Your Reservation")
    st.markdown("Press the button below to record your reservation details for 5 seconds.")

    if st.button("Record"):
        # Record audio for 5 seconds
        st.info("Recording... Please speak now.")
        fs = 44100  # Sample rate
        duration = 6  # Duration in seconds
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
        sd.wait()  # Wait for the recording to finish

        # Save the recorded audio
        filename = "reservation.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())

        st.success(f"Recording saved as `{filename}`.")
        query=pipe("reservation.wav",generate_kwargs={"language": "<|en|>"})
        file_name = "query.txt"
        print(query['text'])
        with open(file_name, 'w') as file:
            file.write(query['text'])
        st.experimental_set_query_params(page="bus_schedules")

# Signup page
def signup():
    st.title("📝 Signup")
    st.markdown("Please fill in your details below to create an account:")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Create a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if new_password == confirm_password:
            # Add registration logic here
            st.success("Account created successfully! Please login.")
        else:
            st.error("Passwords do not match. Please try again.")
import pandas as pd

# Function to display bus schedules
def bus_schedules():
    st.title("🚌 Bus Schedules")
    st.markdown("Below are the available bus schedules:")
    with open("query.txt", 'r') as file:
        reservation = file.read()
    # Sample data for bus schedules
    query=extract_trip_details(reservation)
    with open("sql_query", 'w') as file:
            file.write(query)
    #query = "SELECT start_city, end_city, fare, departure_date, departure_time,capacity FROM schedule,bus where bus.id = schedule.bus_id and start_city='islamabad' and end_city='multan' and departure_date='2024-12-01';"
    connection = connect_to_database()
    schedule_data=None
    if connection:
        schedule_data = fetch_schedule_data(connection,query)
    
    # Convert to DataFrame and display as a table
    df = pd.DataFrame(schedule_data)
    st.table(df)

    # Record button at the end
    number=0
    st.markdown("### Record Additional Details")
    if st.button("Record"):
        st.info("Recording... Please speak now.")
        fs = 44100  # Sample rate
        duration = 5  # Duration in seconds
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
        sd.wait()  # Wait for the recording to finish
        
        # Save the recorded audio
        filename = "bus_schedule_notes.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())

        st.success(f"Recording saved as `{filename}`.")
        translation=pipe("bus_schedule_notes.wav",generate_kwargs={"language": "<|en|>"})
        number=select_schedule_number(translation['text'])
    if st.button("Proceed to Seat Selection"):
            st.experimental_set_query_params(page="select_seats",fare=schedule_data[number-1][2])

# Route to the appropriate page based on query parameters
def route():
    params = st.experimental_get_query_params()
    page = params.get("page", ["main"])[0]

    if page == "login":
        login()
    elif page == "signup":
        signup()
    elif page == "reservation":
        record_voice()
    elif page == "bus_schedules":
        bus_schedules()
    elif page == "select_seats":
        select_seats()
    elif page == "payment":
        payment()
    else:
        main()

# Set page style
set_page_style()

# Route the app
if __name__ == "__main__":
    route()
