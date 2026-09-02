from typing import List
from langchain_core.documents import Document
from tavily import TavilyClient
import config


def tavily_search(query: str, max_results: int = 5) -> List[Document]:
    """
    Perform web search fallback using Tavily when local documents
    lack sufficient evidence. Returns results as Document objects.
    """
    if not config.TAVILY_API_KEY or config.TAVILY_API_KEY == "your_tavily_api_key_here":
        print("[WARN] TAVILY_API_KEY is not configured. Cannot perform web search fallback.", flush=True)
        return []

    print(f"[INFO] Performing Tavily Web Search fallback for: '{query}'...", flush=True)

    try:
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

        documents: List[Document] = []
        for result in response.get("results", []):
            content = result.get("content", "")
            title = result.get("title", "")
            url = result.get("url", "Web Search")

            text = f"Title: {title}\nURL: {url}\n\n{content}"
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "title": title,
                        "is_web": True,
                    },
                )
            )

        print(f"[INFO] Retrieved {len(documents)} results from Tavily Web Search.", flush=True)
        return documents

    except Exception as e:
        print(f"[WARN] Tavily web search failed: {e}", flush=True)
        return []
