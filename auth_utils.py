"""
Simple, self-contained email/password auth backed by MongoDB.

Passwords are hashed with bcrypt before storage -- plaintext passwords
are never written to the database or logged. Session state tracks the
logged-in user for the duration of the browser session (not persisted
across browser restarts; add a cookie-based "remember me" later if needed).
"""
import re
import bcrypt
import streamlit as st

import db_utils as db

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))


def is_logged_in() -> bool:
    return st.session_state.get("user") is not None


def current_user():
    return st.session_state.get("user")


def logout():
    st.session_state.pop("user", None)


def login(email: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    if not email or not password:
        return False, "Please enter both email and password."

    user = db.get_user_by_email(email)
    if not user:
        return False, "No account found with that email."
    if not verify_password(password, user["password_hash"]):
        return False, "Incorrect password."

    st.session_state["user"] = {"email": user["email"], "name": user["name"]}
    return True, "Logged in successfully."


def signup(email: str, name: str, password: str, confirm_password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    if not email or not name or not password:
        return False, "Please fill in all fields."
    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if password != confirm_password:
        return False, "Passwords do not match."

    try:
        db.create_user(email, name, hash_password(password))
    except ValueError as e:
        return False, str(e)

    st.session_state["user"] = {"email": email.lower().strip(), "name": name.strip()}
    return True, "Account created successfully."


def require_login():
    """Call at the top of any protected page. Stops rendering if not logged in."""
    if not is_logged_in():
        st.warning("Please log in from the landing page to access this section.")
        st.stop()
