import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from interpretable_ml_kit import get_german_credit_data, pdp, pdp_pretty_plot, ice, ice_pretty_plot, ale, ale_pretty_plot

# Data
germanCreditData = get_german_credit_data()
x = germanCreditData.drop(["creditRisk"], axis=1)
y = germanCreditData["creditRisk"]


# Model
model_fit = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy', 
                                   max_depth=8, max_features='auto', max_leaf_nodes=None, 
                                   min_impurity_decrease=0.0, min_impurity_split=None, 
                                   min_samples_leaf=9, min_samples_split=9,
                                   min_weight_fraction_leaf=0.0, 
                                   # presort='deprecated',
                                   random_state=42, splitter='best').fit(x, y)

predictions = model_fit.predict(x)


# Features Importance
features_importance = zip(x.columns, model_fit.feature_importances_)
features, importance_score = zip(*features_importance)
feature_importance = pd.DataFrame({"Feature": features, "Score": importance_score})
feature_importance.set_index("Feature", inplace=True)


# App
st.set_page_config(
    page_title = "Interpretable ML | Belgada Zainab",
    page_icon="💎",
    layout="wide"
)

menu_selectbox = st.sidebar.selectbox(
    "How Can I help you?",
    ["---", "Features interpretability", "Features importance", "Show the german credit dataset"]
)


if menu_selectbox == "---":
    st.image("Logo-couleur-MasterESA-RVB.jpg")
    st.markdown(
        """
        # INTERPRETABLE ML PROJECT
        ## By Zainab BELGADA
        ###
        """
    )
    st.markdown(
        """
        In this interactive application you can:  
        - Explore the german credit Dataset.
        - Rank features by their importance.
        - Show PD, ICE, and ALE plots.
        ##  
        Github: https://github.com/belzaina/mlinterp
        """
    )
elif menu_selectbox == "Features interpretability":
    features_selectbox = st.sidebar.selectbox(
        "Please Choose a Feature",
        x.columns.map(lambda x: x if "_" not in x else x.split("_")[0] + "_").unique().tolist(),
        help = "The underscore indicates categorical variables"
    )
    
    pretty_feature_name = features_selectbox[:-1] if "_" in features_selectbox else features_selectbox
    
    reference_feature = [f for f in x.columns if (features_selectbox in f)] if "_" in features_selectbox else features_selectbox
    reference_feature = reference_feature[0] if (type(reference_feature) == list and len(reference_feature) == 1) else reference_feature
    # PDP
    pdp_grid, pdp_values, feature_type = pdp(model_fit, x, reference_feature)
    pdp_chart_data = pd.DataFrame({"x": pdp_grid, "PD": pdp_values})
    # ICE
    ice_grid, ice_values, feature_type = ice(model_fit, x, reference_feature, grid_resolution=50)
    ice_chart_data = pd.DataFrame(ice_values.T)
    ice_chart_data.columns = [str(c) for c in ice_chart_data.columns]
    ice_chart_data["index"] = ice_grid
    # ALE
    ale_grid, ale_values, feature_type = ale(model_fit, x, reference_feature)
    ale_chart_data = pd.DataFrame({"x": ale_grid, "ALE": ale_values})
    col1, col2 = st.beta_columns(2)
    
    with col1:
        st.write(
            alt.Chart(pdp_chart_data, title = "PDP").mark_line().encode(x = alt.X("x", title = pretty_feature_name), y = alt.Y("PD", title = "")).configure_title(fontSize=18).properties(width=450)
        )
    with col2:
        base = alt.Chart(title = "ICE").mark_line().encode(x = alt.X("index", title = pretty_feature_name))
        st.write(
            alt.layer( *[base.encode(y = alt.Y(col, title = "")) for col in ice_chart_data.columns if col!="index"], data = ice_chart_data).configure_title(fontSize=18).properties(width=450)
        )
    
    st.write(
        alt.Chart(ale_chart_data, title = "ALE").mark_line().encode(x = alt.X("x", title = pretty_feature_name), y = alt.Y("ALE", title = "")).configure_title(fontSize=18).properties(width=450)
    )
    
elif menu_selectbox == "Features importance":
    st.header('Features Importance')
    st.subheader('Model: DecisionTreeClassifier')
    st.header('')
    st.bar_chart(feature_importance)
    
else:
    st.header('German Credit Dataset')
    st.subheader('Preprocessed Version')
    st.header('')
    st.dataframe(germanCreditData)


st.sidebar.markdown("#")
st.sidebar.text("Zainab BELGADA\nzainab.belgada@etu.univ-orleans.fr\nhttps://github.com/belzaina")
      
    

