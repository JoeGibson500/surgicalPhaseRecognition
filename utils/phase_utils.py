import re

def clean_phase_name(phase_name):
    """
    Cleans phase names by removing unwanted text like "(partial)" or "(attempted)" or commas.
    
    Args:
    - phase_name (str): The original phase name.
    
    Returns:
    - str: Cleaned phase name or an empty string if input is invalid.
    """
    if not isinstance(phase_name, str):  # Handle NaN, None, or non-string inputs
        return ""

    phase = phase_name.strip().lower()
    phase = re.sub(r"\s*\(attempt\)|\s*\(partial\)", "", phase)
    phase = re.sub(r",", "", phase)
    phase = re.sub(r"though", "through", phase)
    
    return phase


def get_phase_to_index():
    return {
        "unknown": 0,
        "pull through": 1,
        "placing rings": 2,
        "suture pick up": 3,
        "suture pull through": 4,
        "suture tie": 5,
        "uva pick up": 6,
        "uva pull through": 7,
        "uva tie": 8,
        "placing rings 2 arms": 9,
        "1 arm placing": 10,
        "2 arms placing": 11,
        "pull off": 12
    }

