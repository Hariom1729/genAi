# pip install pydantic wikipedia-api requests beautifulsoup4

from pydantic import BaseModel
import wikipediaapi

# Model
class Institution(BaseModel):
    name: str
    summary: str

# Wikipedia setup
wiki = wikipediaapi.Wikipedia(
    user_agent="MyBot/1.0",
    language="en"
)

# Input
name = input("Enter institution name: ")

# Fetch data
page = wiki.page(name)

if page.exists():
    data = Institution(
        name=name,
        summary=page.summary[:200]
    )

    print(data.model_dump_json(indent=2))
else:
    print("Page not found")