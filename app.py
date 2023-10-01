import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    data = pd.read_csv('data/iris.csv')
    return data


def main():
    st.title("Iris Dataset Visualization")
    data = load_data()
    st.sidebar.header("Variety")
    species_to_plot = st.sidebar.selectbox(
        "Select Variety:", data['variety'].unique())
    filtered_data = data[data['variety'] == species_to_plot]
    st.write(f"Displaying data for variety: {species_to_plot}")
    st.write(filtered_data.describe())

    st.subheader("Scatterplot")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal.length', y='sepal.width',
                    data=filtered_data, hue='variety', palette='Set1')
    st.pyplot(plt)

    st.subheader("Histograms")
    plt.figure(figsize=(8, 6))
    sns.histplot(data=filtered_data, x='sepal.length', kde=True)
    st.pyplot(plt)

    st.subheader("Box Plots")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=filtered_data, x='variety', y='sepal.length')
    st.pyplot(plt)

    st.subheader("Pair Plot")
    plt.figure(figsize=(8, 6))
    sns.pairplot(filtered_data, hue='variety', palette='Set1')
    st.pyplot(plt)

    st.subheader("Correlation Heatmap")
    numeric_data = filtered_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    st.subheader("Variety Counts")
    species_counts = filtered_data['variety'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=species_counts.index, y=species_counts.values)
    st.pyplot(plt)


if __name__ == '__main__':
    main()
