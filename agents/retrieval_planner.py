from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class RetrievalPlan(BaseModel):
    """Retrieval strategy decision."""
    strategy: Literal["dense", "sparse", "hybrid", "multi_query"] = Field(
        description="Retrieval strategy to use"
    )


class RetrievalPlanner:
    """Decides retrieval strategy based on query type and content."""

    def __init__(self, model: str = "mistral"):
        self.llm = ChatOllama(model=model)
        self.parser = PydanticOutputParser(pydantic_object=RetrievalPlan)

        self.prompt = ChatPromptTemplate.from_template(
            """Choose the best retrieval strategy:
            - dense: semantic lookup
            - sparse: keyword / exact match
            - hybrid: dense + sparse
            - multi_query: expand query into variants

            Query: {query}
            Route: {route}

            {format_instructions}
            """
        )

    def plan(self, query: str, route: str) -> RetrievalPlan:
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({
            "query": query,
            "route": route,
            "format_instructions": self.parser.get_format_instructions(),
        })
