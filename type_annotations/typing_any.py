from typing import Any

def get_value(data: dict[str, Any], key: str) -> Any:
    """Get the value associated with a key in a dictionary."""
    return data.get(key, None)

print(get_value({"name": "Alice", "age": 30}, "name"))  # Returns: "Alice"

