#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pip
pip.main(["install","streamlit"])


# In[34]:


import pip
pip.main(["install","PIL"])


# In[76]:


import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[77]:


st.header("Diabetes Detection App")


# In[78]:


image=Image.open(r"C:\Users\ASUS\Desktop\123.jpg")


# In[79]:


st.image(image)


# In[80]:


data=pd.read_csv(r"C:\Users\ASUS\Desktop\diabetes.csv")


# In[81]:


data.head()


# In[82]:


st.subheader("Data")


# In[83]:


st.dataframe(data)


# In[84]:


st.subheader("Data Description")


# In[85]:


st.write(data.iloc[:,:8].describe())


# In[86]:


x=data.iloc[:,1:].values
y=data.iloc[:,0].values


# In[87]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[88]:


model=RandomForestClassifier(n_estimators=500)


# In[89]:


model.fit(x_train,y_train)


# In[90]:


y_pred=model.predict(x_test)


# In[91]:


st.subheader("Accuracy of Trained Model")


# In[92]:


st.write(accuracy_score(y_test,y_pred))


# In[93]:


st.subheader("Enter Your Input Data")


# In[94]:


st.text_input(label="Enter your age")


# In[95]:


st.slider("Set your age",0,100,0)


# In[96]:


st.text_area(label="Describe you")


# In[97]:


def user_inputs():
    preg=st.slider("Pregnancy",0,20,0)
    glu=st.slider("Glucose",0,200,0)
    bp=st.slider("Blood Presssure",0,130,0)
    sthick=st.slider("Skin Thickness",0,100,0)
    ins=st.slider("Insulin",0.0,1000.0,0.0)
    bmi=st.slider("BMI",0.0,70.0,0.0)
    dpf=st.slider("DPF",0.000,3.000,0.000)
    age=st.slider("Age",0,100,0)

    input_dict={"Pregnancies":preg,
                "Glucose":glu,
                "Blood Pressure":bp,
                "Skin Thickness":sthick, 
                "Insulin":ins,"BMI":bmi,
                "DPF":dpf,
                "Age":age}
    
    return pd.DataFrame(input_dict,index=["User Input Values"])


# In[98]:


ui=user_inputs()


# In[99]:


st.subheader("Entered Input Data")


# In[100]:


st.write(ui)


# In[101]:


st.subheader("Predictions (0-Don Diabetes, 1-Diabetes)")


# In[102]:


st.write(model.predict(ui))

