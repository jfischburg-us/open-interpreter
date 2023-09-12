"""
This module provides utility functions for working with JSON data,
specifically designed for handling OpenAI streaming responses.

Functions:
- merge_deltas(original, delta): Pushes a delta into the original JSON object,
great for reconstructing complete message objects in streaming responses.
- parse_partial_json(a_str): Attempts to parse a potentially incomplete JSON
string, handling escape characters and unmatched brackets.

Usage:
You can use these functions to efficiently process and manipulate JSON data,
particularly when dealing with partial or streaming data from APIs
like OpenAI's.

Example:
import json_utils

# Merge a delta into an original JSON object
original_data = {"name": "John", "age": 30}
delta_data = {"age": 31, "city": "New York"}
merged_data = json_utils.merge_deltas(original_data, delta_data)

# Parse a potentially incomplete JSON string
partial_json = '{"key": "value", "incomplete": {"nested": true'
parsed_data = json_utils.parse_partial_json(partial_json)

"""
import json


def merge_deltas(original, delta):
    """
    Pushes the delta into the original and returns that.

    Great for reconstructing OpenAI streaming responses
        -> complete message objects.
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


def parse_partial_json(a_str):
    """
    Attempt to parse a potentially incomplete JSON string, handling escape
    characters and unmatched brackets.

    This function takes a string `a_str` that may contain a partial JSON
    object, and attempts to parse it. It processes the string character by
    character to handle situations like unmatched brackets
    and escape sequences.

    Args:
        a_str (str): The input string containing potentially
        incomplete JSON data.

    Returns:
        dict or None: A parsed JSON object if parsing succeeds,
        or None if parsing fails due to malformed input.

    Example:
        partial_json = '{"key": "value", "incomplete": {"nested": true'
        parsed_data = parse_partial_json(partial_json)
        if parsed_data:
            print(parsed_data)
        else:
            print("Parsing failed due to malformed input.")

    Note:
        This function is designed to handle incomplete JSON strings and is
        particularly useful for scenarios where you receive JSON data in a
        streaming fashion, allowing you to process and work with partial data.

    Raises:
        None: This function does not raise any custom exceptions.

    """
    # Attempt to parse the string as-is.
    try:
        return json.loads(a_str)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_s = ""
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in a_str:
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == '\n' and not escaped:
                # Replace the newline character with the escape sequence.
                char = '\\n'
            elif char == '\\':
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '"':
                is_inside_string = True
                escaped = False
            elif char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif char == '}' or char == ']':
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    # Mismatched closing character; the input is malformed.
                    return None

        # Append the processed character to the new string.
        new_s += char

    # If we're still inside a string at the end of processing,
    # we need to close the string.
    if is_inside_string:
        new_s += '"'

    # Close any remaining open structures in the reverse order
    # that they were opened.
    for closing_char in reversed(stack):
        new_s += closing_char

    # Attempt to parse the modified string as JSON.
    try:
        return json.loads(new_s)
    except json.JSONDecodeError:
        # If we still can't parse the string as JSON,
        # return None to indicate failure.
        return None
