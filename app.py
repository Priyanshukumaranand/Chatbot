import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load the data
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

#create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state =0 , max_iter=10000)

#preprocess the data
tags=[]
patterns=[]
for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

#training the model
x= vectorizer.fit_transform(tags)
y= tags
clf.fit(x,y)

def chatbot(input_text):
    input_text =vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
            return response
        return "Cant answer the question"

counter = 0
def main():
    global counter
    st.title("Implementation of chatbot using NLP")

    # create a sidebar menu with options
    menu = ["Home", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home page
    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the home page of the chatbot. You can navigate to the chatbot page to interact with the chatbot.")

        # check if the chat_log.csv file exists , and if it does, display the number of interactions
        if not os.path.exists("chat_log.csv"):
            with open('chat_log.csv', 'w',newline='',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["User Input","Chatbot Response","Timestamp"])
        
        counter+=1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            #convert the user input to lowercase
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            #get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            #save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a',newline='',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([user_input_str,response,timestamp])

            if response.lower() in ['goodbye','bye']:
                st.write("Chatbot: Goodbye! Have a great day!")
                st.stop()
    
    #conversation histry menu

    elif choice == "Conversation History":
        st.subheader("Conversation History")
        with open("chat_log.csv", "r",newline='',encoding='utf-8') as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("")

if __name__ == "__main__":
    main()