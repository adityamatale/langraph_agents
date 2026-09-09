from typing import TypedDict

class Movie(TypedDict):
    title: str  # fields with datatype of field | type safety | readablity
    year: int
    rating: float

movie = Movie(title="Inception", year=2010, rating=8.8)

print(movie)