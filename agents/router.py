from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class Route(BaseModel):
    """Routing decision for user query."""
    route: Literal["factual", "analytical", "policy"] = Field(
        description="Type of the user query"
    )


class RouterAgent:
    """Classifies a user query into a routing category."""

    def __init__(self, model: str = "mistral"):
        self.llm = ChatOllama(model=model)
        self.parser = PydanticOutputParser(pydantic_object=Route)

        self.prompt = ChatPromptTemplate.from_template(
            """Classify the user query into one of:
            - factual
            - analytical
            - policy

            Query: {query}

            {format_instructions}
            """
        )

    def route(self, query: str) -> Route:
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({
            "query": query,
            "format_instructions": self.parser.get_format_instructions(),
        })
