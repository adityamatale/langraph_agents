from typing import Union

def process_input(value: Union[int, str]) -> None:  # type safety | readability
    if isinstance(value, int):
        print(f"Processing an integer: {value}")
    elif isinstance(value, str):
        print(f"Processing a string: {value}")
    else:
        raise ValueError("Unsupported type")

process_input(42)        # Output: Processing an integer: 42
process_input("Hello")   # Output: Processing a string: Hello
process_input(3.14)      # Raises ValueError: Unsupported type