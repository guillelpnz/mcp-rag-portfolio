"""MCP server — exposes the profile RAG as Model Context Protocol tools.

Any MCP-compatible client (Claude Desktop, IDEs, agents) can connect over
stdio and call these tools. Run with: `python -m app.mcp_server`.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .rag import RAGEngine

mcp = FastMCP("guillermo-profile")
_engine = RAGEngine()


@mcp.tool()
def ask_profile(question: str) -> str:
    """Answer a question about Guillermo's CV, experience and skills.

    Use this for anything about his background: roles, tech stack, projects,
    years of experience, or what he worked on at a given company.
    """
    result = _engine.query(question)
    sources = ", ".join(result["sources"]) or "none"
    return f"{result['answer']}\n\n(sources: {sources})"


@mcp.tool()
def list_sources() -> str:
    """List the profile documents currently indexed in the knowledge base."""
    data = _engine.collection.get()
    sources = sorted({m["source"] for m in data.get("metadatas", [])})
    return "\n".join(sources) if sources else "No documents indexed yet."


def main() -> None:
    _engine.ingest()  # index profile docs on startup
    mcp.run()


if __name__ == "__main__":
    main()
