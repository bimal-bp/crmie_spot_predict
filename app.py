import streamlit as st

# Custom CSS for styling
st.markdown("""
    <style>
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #f0f2f6;
        color: #333;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Login Page
def login_page():
    st.title("🔐 Login Page")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120)
    married_status = st.selectbox("Married Status", ["Single", "Married", "Divorced", "Widowed"])
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    
    if st.button("Login"):
        st.session_state['logged_in'] = True
        st.session_state['user_info'] = {
            'name': name,
            'age': age,
            'married_status': married_status,
            'gender': gender
        }
        st.success("Logged in successfully!")
        st.experimental_rerun()

# Option 1 Page
def option_1():
    st.title("📊 Option 1: Crime Risk Analysis")
    st.write("This is the Crime Risk Analysis section.")
    # You can add the crime risk analysis code here

# Option 2 Page
def option_2():
    st.title("📈 Option 2: Another Analysis")
    st.write("This is another analysis section.")
    # You can add another analysis code here

# Main App Logic
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login_page()
    else:
        st.sidebar.title(f"Welcome, {st.session_state['user_info']['name']}!")
        st.sidebar.write(f"Age: {st.session_state['user_info']['age']}")
        st.sidebar.write(f"Married Status: {st.session_state['user_info']['married_status']}")
        st.sidebar.write(f"Gender: {st.session_state['user_info']['gender']}")
        
        option = st.sidebar.radio("Choose an option:", ["Option 1", "Option 2"])
        
        if option == "Option 1":
            option_1()
        elif option == "Option 2":
            option_2()

if __name__ == "__main__":
    main()
