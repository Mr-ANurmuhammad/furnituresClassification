import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# import platform
#
# plt = platform.system()
# if plt == "Linux":pathlib.WindowsPath = pathlib.PosixPath

st.title("Furnitures classification model!")

file =st.file_uploader("Upload image", type=("png", "jpeg", "jpg","svg"))

if file is not None:
    img = PILImage.create(file)
    st.image(file)

    model =load_learner("furnitures_model2.pkl")

    # predict
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik:{probs[pred_id]*100:1f}%")
    # plotting
    fig =px.bar(x=probs, y=model.dls.vocab)
    st.plotly_chart(fig)

else:
    st.warning("Please upload a picture!")