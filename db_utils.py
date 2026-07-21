"""
MongoDB Atlas connection and data-access helpers.

Two collections:
  - users:       {email, name, password_hash, created_at}
  - predictions: {user_email, timestamp, curated_inputs, mean_prob,
                   risk_band, prob_min, prob_max, n_samples, model_name}
"""
import os
from datetime import datetime, timezone

import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi

try:
    from db_config import MONGO_URI
except ImportError:
    MONGO_URI = os.environ.get("MONGO_URI", "")

DB_NAME = "sepsis_risk_app"


@st.cache_resource
def get_client():
    if not MONGO_URI:
        raise RuntimeError(
            "MongoDB connection string not found. Set it in db_config.py "
            "(local) or the MONGO_URI environment variable (deployed)."
        )
    client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
    client.admin.command("ping")  # fail fast with a clear error if unreachable
    return client


def get_db():
    return get_client()[DB_NAME]


# ---------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------
def get_user_by_email(email):
    return get_db().users.find_one({"email": email.lower().strip()})


def create_user(email, name, password_hash):
    db = get_db()
    if get_user_by_email(email):
        raise ValueError("An account with this email already exists.")
    db.users.insert_one({
        "email": email.lower().strip(),
        "name": name.strip(),
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc),
    })


# ---------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------
def save_prediction(user_email, curated_inputs, mean_prob, risk_band,
                     prob_min, prob_max, n_samples, model_name):
    get_db().predictions.insert_one({
        "user_email": user_email.lower().strip(),
        "timestamp": datetime.now(timezone.utc),
        "curated_inputs": curated_inputs,
        "mean_prob": mean_prob,
        "risk_band": risk_band,
        "prob_min": prob_min,
        "prob_max": prob_max,
        "n_samples": n_samples,
        "model_name": model_name,
    })


def get_predictions_for_user(user_email, limit=100):
    cursor = (
        get_db().predictions
        .find({"user_email": user_email.lower().strip()})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def delete_prediction(prediction_id):
    from bson import ObjectId
    get_db().predictions.delete_one({"_id": ObjectId(prediction_id)})
