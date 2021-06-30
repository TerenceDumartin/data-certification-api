
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/predict/")
def predict(acousticness,
            danceability,
            duration_ms,
            energy,
            explicit,
            id,
            instrumentalness,
            key,
            liveness,
            loudness,
            mode,
            name,
            release_date,
            speechiness,
            tempo,
            valence,
            artist):

    X = pd.DataFrame(dict(acousticness=[float(acousticness)],
                          danceability=[float(danceability)],
                          duration_ms=[int(duration_ms)],
                          energy=[float(energy)],
                          explicit=[int(explicit)],
                          id=id,
                          instrumentalness=[float(instrumentalness)],
                          key=[int(key)],
                          liveness=[float(liveness)],
                          loudness=[float(loudness)],
                          mode=[int(mode)],
                          name=name,
                          release_date=release_date,
                          speechiness=[float(speechiness)],
                          tempo=[float(tempo)],
                          valence=[float(valence)],
                          artist=artist))

    # pipeline the model
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(artist=artist, name=name, prediction=pred)
