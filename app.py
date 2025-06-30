# Simple Q&A App using Streamlit

# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")
    
    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!
    
    my_documents = [
        """Physical health forms the foundation of holistic well-being, encompassing the proper functioning of the body through exercise, nutrition, sleep, and preventive care. Maintaining physical health begins with regular physical activity—at least 150 minutes of moderate aerobic exercise per week, as recommended by the World Health Organization. Activities like walking, cycling, or strength training help improve cardiovascular health, boost immunity, and reduce risks of chronic diseases like type 2 diabetes and hypertension. Nutrition also plays a critical role: a balanced diet rich in fruits, vegetables, lean proteins, and whole grains fuels the body and supports cellular repair. Sleep, often underestimated, is essential for physical recovery; adults need 7–9 hours nightly to regulate hormones and support cognitive function. Preventive measures such as annual check-ups, vaccinations, and routine screenings (e.g., mammograms, blood pressure checks) catch potential issues early. Lifestyle choices like avoiding smoking and limiting alcohol are also crucial. One might ask: What is the impact of physical inactivity on long-term health? The answer is stark—physical inactivity contributes to over 3.2 million deaths globally each year. By understanding and practicing good physical habits, individuals can significantly enhance their quality of life and support all other facets of holistic health.
.
""",
        
        """Mental health refers to our cognitive and psychological well-being—how we think, learn, and process information. It's one of the core pillars of holistic health and affects every area of life. According to the World Health Organization, depression affects more than 280 million people globally, making it one of the leading causes of disability. Mental health isn’t just about avoiding illness—it’s about fostering resilience, focus, and a growth mindset. Practices like mindfulness, meditation, and journaling can help reduce anxiety and increase self-awareness. Cognitive Behavioral Therapy (CBT), for example, is an evidence-based method that has helped countless individuals reframe negative thoughts. Even basic routines like getting adequate sleep, maintaining social connections, and spending time in nature contribute significantly to mental clarity. Consider this question: How can I tell if my mental health is declining? Warning signs include persistent sadness, trouble concentrating, irritability, or withdrawal from social activities. Mental health also intersects with physical health; stress can increase the risk of cardiovascular disease and weaken the immune system. Schools, workplaces, and communities can foster mental well-being by promoting open conversations and providing access to resources. Understanding mental health helps individuals take control of their minds, leading to better decisions and a more balanced life.

""",
        
        """Emotional health refers to our ability to understand, express, and manage feelings in a constructive way. It plays a pivotal role in relationships, stress management, and overall happiness. Someone emotionally healthy isn't devoid of negative emotions but rather able to process them without becoming overwhelmed. Emotional health is rooted in self-awareness and emotional intelligence (EQ)—a concept popularized by psychologist Daniel Goleman. EQ includes recognizing one's emotions, managing them, showing empathy, and handling interpersonal relationships judiciously. For example, a person with high EQ may recognize when they’re feeling anxious before a presentation and use deep breathing or positive self-talk to calm themselves. Emotional regulation skills are teachable and often start with techniques such as mindfulness, gratitude journaling, or seeking therapy. According to the American Psychological Association, chronic unmanaged emotional stress can lead to higher levels of cortisol, increasing the risk for heart disease and obesity. Questions like How do I build emotional resilience? are common—and the answer lies in a mix of healthy coping mechanisms, supportive relationships, and professional guidance when necessary. Cultivating emotional health not only enhances inner peace but also enriches personal and professional relationships, helping individuals navigate life's challenges with greater ease and clarity.
""",
        
        """Spiritual health is the aspect of holistic well-being that deals with a sense of purpose, values, and connection—whether to a higher power, nature, or simply to something greater than oneself. While not necessarily tied to religion, spiritual health often involves practices that bring meaning and transcendence. According to a 2020 Pew Research study, over 84% of the global population identifies with a religious group, indicating the significant role spirituality plays worldwide. Common spiritual practices include meditation, prayer, reflection, and participation in community rituals. These activities have been shown to reduce stress, improve emotional stability, and even enhance longevity. For example, Harvard researchers found that regular meditation can change brain structure, increasing gray matter in areas related to memory and compassion. Spiritual health often answers questions like Why am I here? or What is my purpose? Individuals with strong spiritual health often report a higher sense of well-being, greater hope, and a more optimistic outlook. It also provides a moral compass that can guide behavior and decision-making. Whether through religious faith, philosophical inquiry, or quiet contemplation, spiritual health helps individuals align their lives with their deepest values, offering peace and a sense of belonging in the broader tapestry of life.
""",
        
        """Holistic health isn’t complete without examining the social and environmental factors that influence well-being. Social health involves the quality of relationships and the support systems we build. Studies show that people with strong social ties have a 50% greater chance of longevity compared to those who are isolated. Social connections improve mood, reduce stress, and even promote better immune functioning. Conversely, loneliness—declared an epidemic by the U.S. Surgeon General in 2023—can increase the risk of premature death as much as smoking 15 cigarettes a day. Environmental health considers how surroundings—air quality, water, noise, housing, and green spaces—affect physical and mental well-being. For example, prolonged exposure to polluted air is linked to respiratory issues and cognitive decline. Questions like How does my community affect my health? are crucial. People living in underserved neighborhoods often face higher rates of chronic illness due to limited access to healthcare, nutritious food, and safe recreation areas. Urban planning, climate change, and sustainability efforts all tie into environmental health. By creating inclusive, clean, and supportive environments, both socially and ecologically, we can foster communities where every individual has the opportunity to thrive holistically.
"""
    ]
    
    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    
    return collection

def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 1.5:  # 1.5 is similarity threshold - adjust as needed
        return "Sorry, that’s outside my wellness library for now... But I’m here to help with anything within the five dimensions of holistic health."
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=150
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    return answer

# MAIN APP STARTS HERE - This is where we build the user interface

# STREAMLIT BUILDING BLOCK 1: PAGE TITLE
# st.title() creates a large heading at the top of your web page
# The emoji 🤖 makes it more visually appealing
# This appears as the biggest text on your page
st.title("Hello, welcome to Holistica! 🧘🏻‍♀️🌀")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/6203748/pexels-photo-6203748.jpeg?_gl=1*1t0xtpi*_ga*MTgyNjIzNDgzMy4xNzUxMTQyMTY1*_ga_8JE65Q40S6*czE3NTExNDIxNjQkbzEkZzEkdDE3NTExNDQ1MjUkajI0JGwwJGgw");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# STREAMLIT BUILDING BLOCK 2: DESCRIPTIVE TEXT  
# st.write() displays regular text on the page
# Use this for instructions, descriptions, or any text content
# It automatically formats the text nicely

import streamlit as st

st.write("An app that focuses on Holistic Health. Discover a world where healing goes beyond medicine!")
st.markdown("""
<p>This app emphasizes the five dimensions of holistic health:</p>
<div style="line-height: 1.2; margin-top: -7px;">
• Physical 🏃‍♀️  <br>
• Mental 🧠  <br>
• Emotional ❤️‍🩹  <br>
• Spiritual 🧿  <br>
• Social/Environmental 🫂💚 <br>
</div>
""", unsafe_allow_html=True)


# STREAMLIT BUILDING BLOCK 3: FUNCTION CALLS
# We call our function to set up the document database
# This happens every time someone uses the app
collection = setup_documents()

# STREAMLIT BUILDING BLOCK 4: TEXT INPUT BOX
# st.text_input() creates a box where users can type
# - First parameter: Label that appears above the box
# - The text users type gets stored in the 'question' variable
# - Users can click in this box and type their question
question = st.text_input("Ask me anything about these five dimensions 👇🏻")


# STREAMLIT BUILDING BLOCK 5: BUTTON
# st.button() creates a clickable button
# - When clicked, all code inside the 'if' block runs
# - type="primary" makes the button blue and prominent
# - The button text appears on the button itself
if st.button("Find my Answer 🔎"):
    
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    # Check if user actually typed something (not empty)
    if question:
        
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        # st.spinner() shows a rotating animation while code runs
        # - Text inside quotes appears next to the spinner
        # - Everything inside the 'with' block runs while spinner shows
        # - Spinner disappears when the code finishes
        with st.spinner("Stretch, sip, or smile - your answer is loading!"):
            answer = get_answer(collection, question)
        
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        # st.write() can display different types of content
        # - **text** makes text bold (markdown formatting)
        # - First st.write() shows "Answer:" in bold
        # - Second st.write() shows the actual answer
        st.write("**Answer:**")
        st.write(answer)
    
    else:
        # STREAMLIT BUILDING BLOCK 9: SIMPLE MESSAGE
        # This runs if user didn't type a question
        # Reminds them to enter something before clicking
        st.write("Please enter a question!")

# STREAMLIT BUILDING BLOCK 10: EXPANDABLE SECTION
# st.expander() creates a collapsible section
# - Users can click to show/hide the content inside
# - Great for help text, instructions, or extra information
# - Keeps the main interface clean
with st.expander("Having trouble? Click me - we’ve got your back!"):
    st.write("""
    This system answers questions about the five dimensions of Holistic Health:
    1. Physical Health
    2. Mental Health
    3. Emotional Health
    4. Spiritual Health
    5. Social and Environmental aspects of Health
    
    Here are some example questions to get you started:
    - What is the impact of physical inactivity on long-term health?
    - How can I tell if my mental health is declining?
    - How do I build emotional resilience?
    """)

# TO RUN: Save as app.py, then type: streamlit run app.py
