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
    return phase
