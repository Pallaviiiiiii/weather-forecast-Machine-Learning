
# #  python -m streamlit run accuracy.py



import streamlit as st
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


from sklearn.svm import SVC

from matplotlib import dates
from datetime import datetime
from matplotlib import rcParams
from API import owm
from pyowm.commons.exceptions import NotFoundError

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")




st.write("""
# Explore different classifier
""")
dataset_name = st.sidebar.selectbox("Select Dataset",("DailyDelhiClimateTrain","DailyDelhiClimateTest"))
st.write(f"## Name of the Dataset: {dataset_name}")
classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))


def get_dataset(dataset_name):
    if dataset_name == "DailyDelhiClimateTrain":
        data = datasets.load_iris()
    # else:
        # data = datasets.load_DailyDelhiClimateTrain()
    X = data.data
    y = data.target
    return X,y

X, y = get_dataset(dataset_name)


def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    elif classifier_name == "Random Forest":
    # if classifier_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == "SVM":
        classifier = SVC(C=params["C"])
    elif classifier_name == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return classifier

classifier = get_classifier(classifier_name,params)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1234)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

accuracy_score = accuracy_score(y_test, y_pred)

st.write(f"## Classifier = {classifier_name}")
st.write(f"## Accuracy = {accuracy_score}")







def main():


    activities = ["EDA", "Plots"]
    choice = st.sidebar.selectbox("Select Activities", activities)


    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload a Dataset", type=["csv"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Shape"):
                st.write(df.shape)

            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Summary"):
                st.write(df.describe())

            # if st.checkbox("Show Value Counts"):
                # st.write(df.iloc[:, -1].value_counts())

            if st.checkbox("Correlation Plot(Matplotlib)"):
                plt.matshow(df.corr())
                st.pyplot()

            if st.checkbox("Correlation Plot(Seaborn)"):
                st.write(sns.heatmap(df.corr(), annot=True))
                st.pyplot()

            # if st.checkbox("Pie Plot"):
            #     all_columns = df.columns.to_list()
            #     column_to_plot = st.selectbox("Select 1 Column", all_columns)
            #     pie_plot = df[column_to_plot].value_counts().plot.pie(
            #         autopct="%1.1f%%")
            #     st.write(pie_plot)
            #     st.pyplot()

    elif choice == 'Plots':
        st.subheader("Data Visualization")
        data = st.file_uploader("Upload a Dataset", type=["csv"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:, -1].value_counts().plot(kind='bar'))
                st.pyplot()

            # Customizable Plot

            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot", [
                                        "area", "bar",  "hist", "box"])
            selected_columns_names = st.multiselect(
                "Select Columns To Plot", all_columns_names)

            if st.button("Generate Plot"):
                st.success("Generating Customizable Plot of {} for {}".format(
                    type_of_plot, selected_columns_names))

                # Plot By Streamlit
                if type_of_plot == 'area':
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)

                elif type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)

                elif type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)

                # Custom Plot
                elif type_of_plot:
                    cust_plot = df[selected_columns_names].plot(
                        kind=type_of_plot)
                    st.write(cust_plot)
                    st.pyplot()

    # elif choice == 'About':
    #     st.subheader("About")




if __name__ == '__main__':
    main()



































#  python -m streamlit run accuracy.py