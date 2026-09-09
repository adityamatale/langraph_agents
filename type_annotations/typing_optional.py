from typing import Optional

def greet(name: Optional[str] = None) -> str:
    if name is None:
        return "Hello, World!"
    else:
        return f"Hello, {name}!"

greeting = greet("Alice")
print(greeting)  # Output: Hello, Alice!

greeting = greet()
print(greeting)  # Output: Hello, World!