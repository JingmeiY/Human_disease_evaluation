import json
import re
from typing import List, Dict, Any, Union

# TOOD: if not correctly parse, raise error, we then skip the whole entry!!
def parse_json_field(field_value: Union[str, dict, int, float]) -> dict:
    """
    Parse a field that can be either a JSON string, dictionary object, or other types.
    
    Args:
        field_value: The field value that might be a JSON string, dict, or other type
        
    Returns:
        Dictionary representation of the field
    """
    if isinstance(field_value, str):
        try:
            return json.loads(field_value)
        except json.JSONDecodeError:
            # Try to parse as key-value pairs separated by colons
            parsed_dict = parse_string_to_dict(field_value)
            if parsed_dict:
                return parsed_dict
            else:
                # If parsing fails, raise error
                raise ValueError(f"Failed to parse string field: {field_value[:50]}...")
    elif isinstance(field_value, dict):
        return field_value
    else:
        # For any other type (int, float, etc.), raise error
        raise ValueError(f"Unsupported field type: {type(field_value)}")


def parse_string_to_dict(text: str) -> dict:
    """
    Parse a string into a dictionary by looking for key-value patterns.
    Handles patterns like 'key: value', 'key = value', '"key": "value"'
    
    Args:
        text: String to parse
        
    Returns:
        Dictionary of parsed key-value pairs, empty dict if parsing fails
    """
    result = {}
    
    # Pattern 1: Try to find "key": "value" or 'key': 'value' patterns
    json_like_pattern = r'["\']([^"\']+)["\']\s*:\s*["\']([^"\']+)["\']'
    matches = re.findall(json_like_pattern, text)
    for key, value in matches:
        result[key.strip()] = value.strip()
    
    # Pattern 2: Try to find key: value patterns (without quotes)
    if not result:
        colon_pattern = r'([^:\n]+):\s*([^:\n]+?)(?=\n|$|[A-Za-z_]\w*\s*:)'
        matches = re.findall(colon_pattern, text, re.MULTILINE)
        for key, value in matches:
            key = key.strip()
            value = value.strip()
            if key and value:
                # Try to convert value to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        result[key] = value.lower() == 'true'
                    elif value.isdigit():
                        result[key] = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        result[key] = float(value)
                    else:
                        result[key] = value
                except:
                    result[key] = value
    
    return result


def extract_information(data: List[Dict[str, Any]], reasoning_keys: List[str], category_keys: List[str]) -> Dict[str, Any]:
    """
    Extract information from prediction data, maintaining original logic but with JSON string/object compatibility.
    Uses case-insensitive key matching to reduce errors from inconsistent capitalization.
    
    Args:
        data: List of dict entries from your JSON
        reasoning_keys: List of possible reasoning keys to match in the data
        category_keys: List of possible category keys to match in the data
        
    Returns:
        Dictionary containing extracted GT and output data with counts
    """
    gt_categories = []
    gt_reasoning = []
    output_categories = []
    output_reasoning = []

    # Convert all keys to lowercase for case-insensitive matching
    reasoning_keys_lower = [rk.lower() for rk in reasoning_keys]
    category_keys_lower = [ck.lower() for ck in category_keys]

    successful_entries = 0
    skipped_entries = 0

    for i, entry in enumerate(data):
        try:
            # Parse GT field - handle both string and object formats
            gt_raw = entry.get("GT", {})
            gt = parse_json_field(gt_raw)
            
            # Parse output field - handle both string and object formats  
            output_raw = entry.get("output", {})
            output = parse_json_field(output_raw)

            # Only add to evaluation list if both GT and output are correctly parsed
            # 1) Check all GT keys to see if they match any in reasoning_keys or category_keys (case-insensitive)
            for key, value in gt.items():
                key_lower = key.lower()
                if any(rk in key_lower for rk in reasoning_keys_lower):
                    gt_reasoning.append(value)
                elif any(ck in key_lower for ck in category_keys_lower):
                    if isinstance(value, str):
                        gt_categories.append(value.lower())
                    elif isinstance(value, (int, float)):
                        gt_categories.append(int(value))

            # 2) Check all Output keys (case-insensitive)
            for key, value in output.items():
                key_lower = key.lower()
                if any(rk in key_lower for rk in reasoning_keys_lower):
                    output_reasoning.append(value)
                elif any(ck in key_lower for ck in category_keys_lower):
                    if isinstance(value, str):
                        output_categories.append(value.lower())
                    elif isinstance(value, (int, float)):
                        output_categories.append(int(value))
            
            successful_entries += 1
            
        except Exception as e:
            # Skip the whole entry if GT or output is not correctly parsed
            print(f"Skipping entry {i+1}: {str(e)}")
            skipped_entries += 1
            continue

    return {
        "GT_categories": gt_categories,
        "GT_reasoning": gt_reasoning,
        "Output_categories": output_categories,
        "Output_reasoning": output_reasoning,
        "counts": {
            "total_samples": len(data),
            "successful_entries": successful_entries,
            "skipped_entries": skipped_entries,
            "gt_categories_count": len(gt_categories),
            "gt_reasoning_count": len(gt_reasoning),
            "output_categories_count": len(output_categories),
            "output_reasoning_count": len(output_reasoning)
        }
    }


def load_json(json_file: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)
