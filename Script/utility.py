import os
import json
import requests
from typing import List, Optional, Any, Dict, Optional

def is_vllm_available(base_url="http://localhost:8000/v1") -> bool:
    """
    Checks if the vLLM server is available by making an HTTP request.
    Args:
        base_url (str): The base URL of the vLLM API.
    Returns:
        bool: True if vLLM is running, False otherwise.
    """
    health_check_url = f"{base_url}/models"  # vLLM exposes available models at this endpoint
    try:
        response = requests.get(health_check_url, timeout=3)  # timeout
        return response.status_code == 200  # If vLLM responds, it's running
    except requests.RequestException:
        return False  # If there's any error (timeout, connection refused), vLLM is down
    

def load_file_content(file_path):
    """
    Loads and returns the content of a text file.

    Parameters:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise
    except IOError as e:
        print(f"An I/O error occurred while reading the file: {e}")
        raise


def load_jsonl(file_path):
    """
    Load a JSONL (JSON Lines) file and return its contents as a list of dictionaries.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: A list of dictionaries representing the JSONL data.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return data

def load_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json_file(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)