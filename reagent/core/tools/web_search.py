"""Web search tool using DuckDuckGo."""
from typing import Any, Dict, List
from strands import tool
import logging

logger = logging.getLogger(__name__)


@tool
def duckduckgo_search(
    query: str,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Dictionary containing search results with titles, snippets, and URLs
        
    Example:
        result = duckduckgo_search("python async programming", max_results=3)
        for item in result['results']:
            print(f"{item['title']}: {item['url']}")
    """
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for i, result in enumerate(ddgs.text(query, max_results=max_results)):
                results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', ''),
                    'rank': i + 1
                })
        
        logger.info(f"DuckDuckGo search for '{query}' returned {len(results)} results")
        
        return {
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        }
        
    except ImportError:
        error_msg = "duckduckgo-search not installed. Run: pip install duckduckgo-search"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'results': []
        }
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'results': []
        }