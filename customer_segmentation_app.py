"""
@author: liz.lungania

"""
    
import pickle
import numpy as np
import streamlit as st
#from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)

@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(total_sales, products_ordered, recency):  
    total_sales = np.log(total_sales)
    products_ordered = np.log(products_ordered)
   
   
     # Making predictions 
    prediction = classifier.predict(
        [[total_sales, products_ordered, recency]])
    return prediction


# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:lightgreen;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Customer Segmentation ML App</h1> 
    </div> 
    """
    
    
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    total_sales = st.number_input("Customers Total Sales") 
    products_ordered = st.number_input("Total Products Ordered")
    recency = st.number_input("Days since last purchase")
    result =""
    
     # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(total_sales, products_ordered, recency) 
        st.success('customer falls under segment {}'.format(result))
        
    
    
if __name__=='__main__': 
     main() 
    
    
    