import time
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp_c": 31, "rain": False}
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b
def save_note(text: str) -> str:
    """Save a note and return its id."""
    return "note-123"
def slow_lookup(q: str) -> str:
    """Look up a fact (slow on purpose)."""
    time.sleep(30) 
    return f"Fact about {q}: It is verified."
def flaky_db(user_id: int) -> dict:
    """Fetch a user record (raises on purpose)."""
    raise ConnectionError("Database connection lost unexpectedly.")