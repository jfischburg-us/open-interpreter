"""
This module provides utility functions for working with
JSON data and merging JSON deltas.

Key Functions:
- merge_deltas(original, delta): Merges a JSON delta
into an original JSON object.
- parse_partial_json(s): Attempts to parse a partial
JSON string and corrects some common issues.
"""
import json


def merge_deltas(original, delta):
    """
    Merge a JSON delta into an original JSON object.

    This function recursively applies a JSON delta (a dictionary)
    into an original JSON object.

    It is particularly useful for reconstructing OpenAI streaming
    responses into complete message objects.

    Args:
        original (dict): The original JSON object to be updated.
        delta (dict): The JSON delta to be merged into the original.

    Returns:
        dict: The original JSON object with the delta applied.
    """
    for key, value in delta.items():
        if isinstance(value, dict):
            if key not in original:
                original[key] = value
            else:
                merge_deltas(original[key], value)
        else:
            if key in original:
                original[key] += value
            else:
                original[key] = value
    return original


def parse_partial_json(partial_json):
    """
    Attempt to parse a partial JSON string and correct common issues.

    This function attempts to parse a partial JSON string and corrects
    some common issues, such as missing closing brackets, newline characters,
    and escaped characters within strings.

    Args:
        s (str): The partial JSON string to be parsed.

    Returns:
        dict or None: The parsed JSON object if successful,
        or None if parsing fails.
    """
    # Attempt to parse the string as-is.
    try:
        return json.loads(partial_json)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_s = ""
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in partial_json:
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                char = "\\n"  # Replace the newline character with the escape sequence.
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '"':
                is_inside_string = True
                escaped = False
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char == "}" or char == "]":
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    # Mismatched closing character; the input is malformed.
                    return None

        # Append the processed character to the new string.
        new_s += char

    # If we're still inside a string at the end of processing, we need to close the string.
    if is_inside_string:
        new_s += '"'

    # Close any remaining open structures in the reverse order that they were opened.
    for closing_char in reversed(stack):
        new_s += closing_char

    # Attempt to parse the modified string as JSON.
    try:
        return json.loads(new_s)
    except json.JSONDecodeError:
        # If we still can't parse the string as JSON, return None to indicate failure.
        return None
