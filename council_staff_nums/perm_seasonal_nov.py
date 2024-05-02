import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import math


@st.cache_data
def load_data():
    data = pd.read_csv(r"postcode_type_count.csv")
    # Remove rowsn where Longitude or Latitude is NaN
    data = data.dropna(subset=["Longitude", "Latitude"])
    data["scale"] = data["count"].apply(lambda count: math.sqrt(count))
    return data


@st.cache_data
def get_perms(df):
    df_perm = df[df["STATUS4"] == "Permanent"]
    return df_perm


@st.cache_data
def get_seasonals(df):
    df_seasonal = df[df["STATUS4"] == "Seasonal"]
    return df_seasonal


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


st.title("Permanent Seasonal Staff Locations")
st.write("### Map Overview")

df = load_data()

midpoint = (np.average(df["Latitude"]), np.average(df["Longitude"]))

df_perm = get_perms(df)
df_seasonal = get_seasonals(df)

ALL_LAYERS = {
    "Permanent": pdk.Layer(
        "ScatterplotLayer",
        data=df_perm,
        opacity=0.2,
        pickable=True,
        stroked=True,
        filled=True,
        radius_scale=20,
        radius_min_pixels=2,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius="scale",
        # get_radius="count",
        get_fill_color=hex_to_rgb("#5F0688"),
        get_line_color=[0, 0, 0],
    ),
    "Seasonal": pdk.Layer(
        "ScatterplotLayer",
        data=df_seasonal,
        opacity=0.2,
        pickable=True,
        stroked=True,
        filled=True,
        radius_scale=20,
        radius_min_pixels=2,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius="scale",
        # get_radius="count",
        get_fill_color=hex_to_rgb("#0BB5FF"),
        get_line_color=[0, 0, 0],
    ),
}

st.sidebar.markdown("### Map Layers")
selected_layers = [
    layer
    for layer_name, layer in ALL_LAYERS.items()
    if st.sidebar.checkbox(layer_name, True)
]


# Set the viewport location
view_state = pdk.ViewState(
    # latitude=52.596462,
    # longitude=-2.157917,
    latitude=midpoint[0],
    longitude=midpoint[1],
    zoom=5,
    bearing=0,
    pitch=0,
)

if selected_layers:
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=selected_layers,
            tooltip={"text": "{Postcode}\n{count}"},
        )
    )
else:
    st.error("Please choose at least one layer above.")

if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write("Not available for this dataset.")
