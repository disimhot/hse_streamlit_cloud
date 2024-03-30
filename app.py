import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# constants
TRAIN = 'data/Training.csv'
TEST = 'data/Testing.csv'

st.balloons()
# functions
@st.cache_data
def load_data(url):
    return pd.read_csv(url)


# train dataframe
df_train = load_data(TRAIN)
X_train = df_train.drop(columns=['Outcome'])
y_train = df_train['Outcome']

#test data
test = load_data(TEST)
X_test = test.drop(columns=['Outcome'])
y_test = test['Outcome']


st.title("Let's predict diabetes")
st.sidebar.title("About")
st.sidebar.info(
    """
    Learning project to diagnose diabetes.
    Train dataset is used for training algorithm and
    it is tested on test dataset.
    """
)
st.text('Choose algorithm to make prediction to diagnose diabetes')
# Multi select
select_alg = st.radio("Algorithms", ["Logistic Regression", "SVC", "DecisionTreeClassifier"])
pressed = st.button("Train")


train_score = 0

if pressed:
    if select_alg == 'Logistic Regression':
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        train_score = lr.score(X_train, y_train)
        y_preds = lr.predict(X_test)
        y_score = accuracy_score(y_test, y_preds)
        st.text(f'Accuracy score is {y_score}')
    elif select_alg == 'SVC':
        svc = SVC(kernel='rbf')
        params_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': ['auto', 'scale'],
        }
        gscv = GridSearchCV(svc, param_grid=params_grid, cv=5)
        gscv.fit(X_train, y_train)
        train_score = gscv.score(X_train, y_train)
        y_preds2 = gscv.predict(X_test)
        y_score = accuracy_score(y_test, y_preds2)
        st.text(f'Accuracy score is {y_score}')
    elif select_alg == 'DecisionTreeClassifier':
        clf = DecisionTreeClassifier()
        clf_params = {
            'criterion': ['gini', 'log_loss'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 5, 15]
        }
        clf_gs = GridSearchCV(clf, param_grid=clf_params, cv=5)
        clf_gs.fit(X_train, y_train)
        clf_gs.score(X_train, y_train)
        decision_preds = clf_gs.predict(X_test)
        y_score = accuracy_score(y_test, decision_preds)
        st.text(f'Accuracy score is {y_score}')


percentage = test['Outcome'].value_counts(normalize=True) * 100

# Plotting the percentage of each class
fig = plt.figure(figsize=(8, 5))
sns.set_palette(['cyan', 'red'])
ax = sns.barplot(x=percentage.index, y=percentage )
plt.title('Percentage of Diabetic and Non Diabetic')
plt.xlabel('Outcome')
plt.ylabel('Percentage (%)')
plt.xticks(ticks=[0, 1], labels=['Non Diabetic', 'Diabetic'])
plt.yticks(ticks=range(0,80,10))

# Displaying the percentage on the bars
for i, p in enumerate(percentage):
    ax.text(i, p + 0.5, f'{p:.2f}%', ha='center', va='bottom')

st.pyplot(fig)







