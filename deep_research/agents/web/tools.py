"""Web search tools: Tavily, Exa, Desearch (Bittensor SN22)."""

import logging
from typing_extensions import Annotated, Literal, Optional
from langchain_core.tools import tool, InjectedToolArg

from deep_research.utils import (
    tavily_search_multiple,
    deduplicate_search_results,
    process_search_results,
    format_search_output,
    _exa_search_multiple,
    desearch_search_multiple,
)

logger = logging.getLogger(__name__)


def _format_web_search(summarized_results: dict) -> dict:
    """Format web search results into {display, items} for index-based reads.

    Returns the display string (via format_search_output) plus an items list
    so the engine can register an [S#] handle and the agent can read by
    index+ref and save via batch_save_selected. Items use the URL as their id.
    """
    display = format_search_output(summarized_results)
    items = [
        {"id": url, "title": r.get("title", url), "url": url,
         "full_text": r.get("full_text", "")}
        for url, r in summarized_results.items()
    ]
    if items:
        display += (
            "\n\nRead a source with fetch_urls(urls=[...]), then save it with "
            "batch_save_selected(items=[{\"ref\": \"S#\", \"index\": N, \"reason\": \"...\"}])."
        )
    return {"display": display, "items": items}


@tool(parse_docstring=True)
async def tavily_search(
    query: str,
    max_results: int = 5,
    topic: Annotated[Literal["general", "news", "finance", "environment", "technology"], InjectedToolArg] = "general",
    days: Annotated[Optional[int], InjectedToolArg] = None,
) -> str:
    """Search the web with Tavily.

    Returns results with snippets/content so you can judge relevance. Read a
    source in full with fetch_urls(urls=[...]) when you want depth, then save
    it with batch_save_selected(items=[{"ref": "S#", "index": N, "reason": "..."}]).

    Use it strategically:
    - Run MULTIPLE searches in parallel for different angles, then read the most
      promising hits before searching again.
    - topic narrows to a domain: 'news' for current events, 'finance' for market
      data, 'technology', etc. Use 'general' when unsure.
    - days restricts to recency — use for time-sensitive queries or fast-moving
      markets; leave unset for evergreen questions.
    - Each call counts toward your SEARCH budget; don't re-search once capped.

    Args:
        query: A single search query to execute
        max_results: Number of results to return (default 6). Can be increased up to 20 if needed.
        topic: Topic to filter results by ('general', 'news', 'finance')
        days: Limit results to the last N days. Use this for time-sensitive queries or market data.
    """
    import time
    search_start = time.time()
    max_results = min(max(max_results, 1), 20)

    logger.info(f"Tavily search: '{query}'")
    search_results = await tavily_search_multiple(
        [query],
        max_results=max_results,
        topic=topic,
        days=days,
        include_raw_content=True,
    )

    unique_results = deduplicate_search_results(search_results)
    summarized_results = await process_search_results(unique_results, tool_name="tavily")

    search_elapsed = time.time() - search_start
    logger.info(f"Tavily complete for '{query}' in {search_elapsed:.2f}s ({len(summarized_results)} sources)")
    return _format_web_search(summarized_results)


@tool(parse_docstring=True)
async def exa_deep_search(
    query: str,
    max_results: int = 10,
) -> str:
    """Run an Exa deep web search for broad, high-recall exploration.

    Use this to map a whole topic — it returns many results per query with
    snippets/content. Read full text via fetch_urls when you want depth, then
    save with batch_save_selected.

    - Run parallel searches ONLY when the research topic explicitly demands 2+
      distinct, non-overlapping angles; otherwise search once and read. It is
      cheaper to use another iteration for a second search than to batch them
      up front.
    - Each call counts toward your SEARCH budget; don't re-search once capped.

    Args:
        query: Search query for Exa deep search
        max_results: Desired number of results (1-20). 5-10 is enough for 90% of
            searches. Only request >10 (up to 20) if the prior search's results
            were clearly insufficient — state why. Never exceed 10 by default.
    """
    import time

    start_time = time.time()
    max_results = min(max(max_results, 1), 20)
    logger.info(f"Exa deep search: '{query}'")

    search_results = await _exa_search_multiple([query], max_results=max_results)
    unique_results = deduplicate_search_results(search_results)
    summarized_results = await process_search_results(unique_results, tool_name="exa")

    elapsed = time.time() - start_time
    logger.info(f"Exa complete for '{query}' in {elapsed:.2f}s ({len(summarized_results)} sources)")
    return _format_web_search(summarized_results)


@tool(parse_docstring=True)
async def desearch_search(
    query: str,
    max_results: int = 10,
    tools: Literal["web+twitter+reddit", "web", "web+arxiv", "web+reddit",
                  "web+youtube", "all"] = "web+twitter+reddit",
    date_filter: Optional[Literal["PAST_24_HOURS", "PAST_2_DAYS", "PAST_WEEK",
                                  "PAST_2_WEEKS", "PAST_MONTH", "PAST_2_MONTHS",
                                  "PAST_YEAR", "PAST_2_YEARS"]] = None,
) -> str:
    """Search web + social + academic sources with Desearch (Bittensor SN22).

    Broader than a plain web search — one query fans out across web pages,
    X/Twitter posts, and Reddit threads (plus arXiv, YouTube, Hacker News and
    Wikipedia via `tools`), which is where real-time sentiment and community
    discussion live. Results carry an [S#] handle — read promising sources with
    fetch_urls(urls=[...]) and save them with batch_save_selected.

    - Run parallel searches ONLY when the topic demands 2+ distinct,
      non-overlapping angles; otherwise search once and read.
    - `tools` picks the source mix: 'web+twitter+reddit' (default) for broad
      coverage, 'web+arxiv' for technical topics, 'web+youtube' for video-led
      topics, 'all' for everything.
    - `date_filter` bounds recency ('PAST_WEEK', 'PAST_MONTH', ...); leave
      unset for evergreen questions.
    - Each call counts toward your SEARCH budget; don't re-search once capped.

    Args:
        query: Search query for Desearch
        max_results: Desired number of results (1-20). 5-10 is enough for most
            searches.
        tools: Source mix preset (see above)
        date_filter: Recency window preset (see above)
    """
    import time

    start_time = time.time()
    max_results = min(max(max_results, 1), 20)
    tool_map = {
        "web+twitter+reddit": ["web", "twitter", "reddit"],
        "web": ["web"],
        "web+arxiv": ["web", "arxiv"],
        "web+reddit": ["web", "reddit"],
        "web+youtube": ["web", "youtube"],
        "all": ["web", "twitter", "reddit", "arxiv", "youtube", "hackernews", "wikipedia"],
    }
    logger.info(f"Desearch search: '{query}' (tools={tools})")

    search_results = await desearch_search_multiple(
        [query], max_results=max_results,
        tools=tool_map.get(tools, ["web", "twitter", "reddit"]),
        date_filter=date_filter,
    )
    unique_results = deduplicate_search_results(search_results)
    summarized_results = await process_search_results(unique_results, tool_name="desearch")

    elapsed = time.time() - start_time
    logger.info(f"Desearch complete for '{query}' in {elapsed:.2f}s ({len(summarized_results)} sources)")
    return _format_web_search(summarized_results)
