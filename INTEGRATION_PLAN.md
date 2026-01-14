# Integration Plan: Enhanced Memory for ReAgent using Local FAISS Vector Store

## Executive Summary

This document outlines a comprehensive plan to integrate the **local_faiss_mcp** vector store as a semantic memory backend for the **ReAgent** reactive agentic orchestration framework. The integration addresses critical limitations in ReAgent's current memory implementation by:

1. **Replacing keyword-based similarity** with vector embeddings for semantic search
2. **Implementing the SHARED tier** via MCP server mode, enabling multiple ReAgent instances to share a common semantic memory

**Key Insight:** The SEMANTIC tier, when deployed as an MCP server, fulfills both the semantic search requirement AND the distributed SHARED tier requirement in a single solution.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution Architecture](#3-proposed-solution-architecture)
4. [Implementation Plan](#4-implementation-plan)
5. [Technical Specifications](#5-technical-specifications)
6. [Alternative Approaches Considered](#6-alternative-approaches-considered)
7. [References](#7-references)

---

## 1. Current State Analysis

### 1.1 ReAgent Framework

**Repository:** https://github.com/nonatofabio/ReAgent

ReAgent is a reactive multi-agent orchestration layer built on top of the AWS Strands Agents SDK. It implements an LLM-first architecture where all decision-making uses Claude (via Bedrock).

#### Core Components (Key Files):

| Component | File Path | Description |
|-----------|-----------|-------------|
| Orchestrator | `/ReAgent/reagent/core/orchestrator.py` (733 lines) | Main swarm execution and agent coordination |
| Memory System | `/ReAgent/reagent/core/memory.py` (587 lines) | Tiered memory with async operations |
| Adaptation Engine | `/ReAgent/reagent/core/adaptation.py` (488 lines) | Real-time performance monitoring |
| MCP Utilities | `/ReAgent/reagent/utils/mcp.py` (128 lines) | MCP configuration and loading |

#### Current Memory Architecture:

```
ReactiveSharedMemory
├── local_cache (dict)       # In-memory entries (~1ms access)
├── persistent_store (dict)  # Disk-backed via JSON (~10ms access)
├── shared_store (dict)      # PLACEHOLDER - NOT IMPLEMENTED (~50ms target)
├── archive_store (dict)     # Long-term archival
└── access_patterns (dict)   # Tracks access history
```

**Configuration Constants** (memory.py:68-71):
- `max_local_entries = 1000`
- `max_persistent_entries = 10000`
- `tier_promotion_threshold = 5`
- `tier_demotion_age = 3600` seconds

### 1.2 Local FAISS MCP Server

**Repository:** https://github.com/nonatofabio/local_faiss_mcp

A Model Context Protocol (MCP) server providing local vector database functionality using FAISS for RAG applications.

#### Core Components (Key Files):

| Component | File Path | Description |
|-----------|-----------|-------------|
| FAISSVectorStore | `/local_faiss_mcp/local_faiss_mcp/server.py:24-174` | Core vector operations class |
| MCP Server | `/local_faiss_mcp/local_faiss_mcp/server.py:177-489` | MCP protocol implementation |
| Document Parser | `/local_faiss_mcp/local_faiss_mcp/document_parser.py` | Multi-format document extraction |
| CLI Interface | `/local_faiss_mcp/local_faiss_mcp/cli.py` | Standalone command-line tool |

#### Capabilities:

| Feature | Details |
|---------|---------|
| Vector Operations | Ingestion, similarity search, re-ranking |
| Embedding Models | `all-MiniLM-L6-v2` (default), any HuggingFace sentence-transformer |
| Persistence | `{index_dir}/faiss.index` + `{index_dir}/metadata.json` |
| MCP Tools | `ingest_document`, `query_rag_store` |
| Document Formats | TXT, MD, PDF (native), DOCX, HTML, EPUB (via Pandoc) |

---

## 2. Problem Statement

### 2.1 Critical Limitations in ReAgent Memory

Based on analysis of `/ReAgent/reagent/core/memory.py`:

#### Problem 1: Keyword-Based Similarity (Not Semantic)
- **Location:** Lines 204-236 (`get_similar_executions()` method)
- **Issue:** Uses Jaccard index on keywords extracted from text
- **Impact:** Cannot find conceptually similar executions across domains

```python
# Current implementation (memory.py:222-226)
common = task_keywords & entry_keywords
if common:
    similarity = len(common) / len(task_keywords | entry_keywords)
```

#### Problem 2: SHARED Tier Not Implemented
- **Location:** Line 550-551
- **Issue:** Mock operation only - logs but doesn't persist to distributed storage
- **Code Comment:** `# TODO: investigate distributed storage options like Redis, etc.`

```python
# Current state (memory.py:546-551)
async def _sync_to_shared_storage(self, key: str, entry: MemoryEntry) -> None:
    """Sync to shared storage (placeholder for distributed storage)."""
    # This would sync to a distributed storage system
    logger.debug(f"Would sync to shared storage: {key}")
```

**Solution:** The SEMANTIC tier via MCP server mode solves this - a single MCP server process provides shared access for multiple ReAgent instances, eliminating the need for separate Redis/distributed storage infrastructure.

#### Problem 3: Linear Search Performance
- No indexing mechanism for memory entries
- O(n) search across all entries
- Performance degrades with large memory stores

#### Problem 4: No Vector Embeddings
- No semantic understanding of task descriptions
- Cannot leverage pre-trained language models for memory
- Missing cross-domain reasoning capabilities

### 2.2 Opportunities Identified

From README.md in ReAgent:
> "TODO: Implement semantic memory tier"

This aligns perfectly with integrating local_faiss_mcp as a semantic memory backend.

---

## 3. Proposed Solution Architecture

### 3.1 High-Level Architecture

#### Single-Agent Mode (Direct Library)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ReAgent Instance                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    ReactiveSharedMemory (Enhanced)                       ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ ││
│  │  │    LOCAL     │  │  PERSISTENT  │  │   SEMANTIC   │  │   ARCHIVE    │ ││
│  │  │   (dict)     │  │   (JSON)     │  │  (replaces   │  │   (JSON)     │ ││
│  │  │   ~1ms       │  │   ~10ms      │  │   SHARED)    │  │   ~100ms     │ ││
│  │  └──────────────┘  └──────────────┘  └──────┬───────┘  └──────────────┘ ││
│  │                                             │                            ││
│  │                                             ▼                            ││
│  │                              ┌──────────────────────────┐                ││
│  │                              │  SemanticMemoryAdapter   │                ││
│  │                              │  (Direct Library Mode)   │                ││
│  │                              └────────────┬─────────────┘                ││
│  └───────────────────────────────────────────│──────────────────────────────┘│
│                                              ▼                               │
│                              ┌──────────────────────────┐                    │
│                              │   FAISSVectorStore       │                    │
│                              │   (in-process)           │                    │
│                              └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Multi-Agent Mode (MCP Server = SHARED Tier)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SHARED SEMANTIC MEMORY (MCP Server)                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    local-faiss-mcp (Standalone Process)                  ││
│  │                                                                          ││
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐││
│  │  │  FAISS Index      │  │  Metadata Store   │  │  Embedding Model      │││
│  │  │  (faiss.index)    │  │  (metadata.json)  │  │  (all-MiniLM-L6-v2)   │││
│  │  └───────────────────┘  └───────────────────┘  └───────────────────────┘││
│  │                                                                          ││
│  │  MCP Tools: ingest_document, query_rag_store                            ││
│  │  Transport: stdio (via subprocess) or SSE (network)                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    ▲                                         │
│              ┌─────────────────────┼─────────────────────┐                   │
│              │                     │                     │                   │
└──────────────│─────────────────────│─────────────────────│───────────────────┘
               │ MCP                 │ MCP                 │ MCP
               │                     │                     │
┌──────────────▼──────┐  ┌──────────▼──────────┐  ┌──────▼──────────────────┐
│  ReAgent Instance 1 │  │  ReAgent Instance 2 │  │  ReAgent Instance N    │
│  ┌────────────────┐ │  │  ┌────────────────┐ │  │  ┌────────────────────┐│
│  │ LOCAL (dict)   │ │  │  │ LOCAL (dict)   │ │  │  │ LOCAL (dict)       ││
│  │ PERSISTENT     │ │  │  │ PERSISTENT     │ │  │  │ PERSISTENT         ││
│  │ SEMANTIC ──────┼─┼──┼──┼─► MCP Client ──┼─┼──┼──┼─► (shared via MCP) ││
│  │ ARCHIVE        │ │  │  │ ARCHIVE        │ │  │  │ ARCHIVE            ││
│  └────────────────┘ │  │  └────────────────┘ │  │  └────────────────────┘│
└─────────────────────┘  └────────────────────┘  └─────────────────────────┘
```

**Key Architecture Decision:** The SEMANTIC tier operates in two modes:
- **Direct mode:** In-process FAISSVectorStore (single agent, best performance)
- **MCP mode:** Connects to shared MCP server (multi-agent, fulfills SHARED tier requirement)

### 3.2 Integration Options

#### Option A: MCP Server Integration (SHARED Tier Solution)

**Use Case:** Multiple ReAgent instances need to share semantic memory

**Pros:**
- **Solves SHARED tier requirement** - single memory store for all agents
- Standard protocol for tool access
- ReAgent already has MCP support (`reagent/utils/mcp.py`)
- Enables distributed deployment
- Clean separation of concerns
- Memory persists independently of agent lifecycle

**Cons:**
- IPC overhead (~5-10ms per call)
- Requires MCP server process management

**Configuration:**
```json
{
  "mcpServers": {
    "shared-semantic-memory": {
      "command": "local-faiss-mcp",
      "args": [
        "--index-dir", "./reagent_shared_memory",
        "--embed", "all-MiniLM-L6-v2"
      ]
    }
  }
}
```

#### Option B: Direct Library Integration (Single-Agent Mode)

**Use Case:** Single ReAgent instance, maximum performance

**Pros:**
- Zero IPC overhead
- Direct Python API access
- Simpler deployment (single process)
- Full control over embedding pipeline

**Cons:**
- No memory sharing between agents
- Memory tied to agent process lifecycle

**Implementation:**
```python
from local_faiss_mcp.server import FAISSVectorStore

class SemanticMemoryAdapter:
    def __init__(self, index_dir: str, embed_model: str = "all-MiniLM-L6-v2"):
        self.store = FAISSVectorStore(
            index_dir=index_dir,
            embed_model_name=embed_model
        )
```

### 3.3 Recommended Approach: Unified SEMANTIC/SHARED Tier

The SEMANTIC tier **replaces** the original SHARED tier concept by using MCP server mode:

| Deployment | Mode | SHARED Tier Status |
|------------|------|-------------------|
| Single agent | Direct library | N/A (not needed) |
| Multi-agent | MCP server | **Implemented via MCP** |

**Configuration-driven selection:**
```python
# Single agent (direct mode - no sharing needed)
orchestrator = ReactiveSwarmOrchestrator(
    semantic_memory_enabled=True,
    semantic_use_mcp=False,  # Direct library
)

# Multi-agent (MCP mode - enables sharing)
orchestrator = ReactiveSwarmOrchestrator(
    semantic_memory_enabled=True,
    semantic_use_mcp=True,   # Connect to shared MCP server
    mcp_config_path="./mcp_config.json"
)
```

**Why this solves SHARED tier:**
1. MCP server runs as independent process
2. Multiple ReAgent instances connect to same server
3. All semantic memory operations go through shared FAISS index
4. No need for Redis/external distributed storage

---

## 4. Implementation Plan

### Phase 1: Foundation (Estimated: 2-3 days)

#### 4.1.1 Add Semantic Memory Tier

**File to modify:** `/ReAgent/reagent/core/memory.py`

**Changes:**

1. Update `MemoryTier` enum to replace SHARED with SEMANTIC (line ~42):
```python
class MemoryTier(Enum):
    LOCAL = "local"
    PERSISTENT = "persistent"
    SEMANTIC = "semantic"      # REPLACES SHARED - provides both semantic search AND sharing via MCP
    ARCHIVE = "archive"
    # SHARED = "shared"        # DEPRECATED - functionality merged into SEMANTIC tier
```

2. Add semantic store initialization (after line 65):
```python
self.semantic_store: Optional[SemanticMemoryAdapter] = None
self.semantic_enabled: bool = semantic_config is not None
if semantic_config:
    self.semantic_store = SemanticMemoryAdapter(**semantic_config)
```

3. Update `_store_in_tier()` method to handle SEMANTIC tier

4. Update `_retrieve_from_tier()` method for semantic retrieval

#### 4.1.2 Create SemanticMemoryAdapter

**New file:** `/ReAgent/reagent/core/semantic_memory.py`

```python
"""
Semantic Memory Adapter for ReAgent
Integrates local_faiss_mcp as vector store backend
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
import os

@dataclass
class SemanticSearchResult:
    """Result from semantic memory search."""
    key: str
    content: str
    distance: float
    metadata: Dict[str, Any]
    source: str

class SemanticMemoryAdapter:
    """
    Adapter for semantic memory using local FAISS vector store.

    Provides:
    - Vector embedding of memory entries
    - Semantic similarity search
    - Automatic index management
    - **SHARED tier functionality via MCP server mode**

    Modes:
    - Direct mode: In-process FAISSVectorStore (single agent)
    - MCP mode: Connect to shared MCP server (multi-agent, implements SHARED tier)
    """

    def __init__(
        self,
        index_dir: str = "./reagent_semantic_memory",
        embed_model: str = "all-MiniLM-L6-v2",
        use_mcp: bool = False,
        mcp_config: Optional[Dict] = None
    ):
        self.index_dir = index_dir
        self.embed_model = embed_model
        self.use_mcp = use_mcp
        self.mcp_client = None

        if use_mcp:
            self._init_mcp_client(mcp_config)
        else:
            self._init_direct_store()

    def _init_direct_store(self):
        """Initialize direct FAISS store connection (single-agent mode)."""
        from local_faiss_mcp.server import FAISSVectorStore
        self.store = FAISSVectorStore(
            index_dir=self.index_dir,
            embed_model_name=self.embed_model
        )

    def _init_mcp_client(self, mcp_config: Optional[Dict]):
        """
        Initialize MCP client for shared memory access (multi-agent mode).

        This enables the SHARED tier functionality - multiple ReAgent instances
        can connect to the same MCP server and share semantic memory.
        """
        from reagent.utils.mcp import MCPClient

        if mcp_config is None:
            mcp_config = {
                "command": "local-faiss-mcp",
                "args": ["--index-dir", self.index_dir, "--embed", self.embed_model]
            }

        self.mcp_client = MCPClient(
            server_name="shared-semantic-memory",
            config=mcp_config
        )

    async def _store_via_mcp(self, document: str, source: str) -> bool:
        """Store document via MCP server (SHARED mode)."""
        result = await self.mcp_client.call_tool(
            "ingest_document",
            {
                "content": document,
                "source": source
            }
        )
        return result.get("chunks_indexed", 0) > 0

    async def _search_via_mcp(self, query: str, top_k: int) -> List[Dict]:
        """Search via MCP server (SHARED mode)."""
        result = await self.mcp_client.call_tool(
            "query_rag_store",
            {
                "query": query,
                "top_k": top_k
            }
        )
        return result.get("results", [])

    async def store(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Store content in semantic memory with embeddings."""
        source = f"memory:{key}"
        document = self._format_for_storage(key, content, metadata)

        if self.use_mcp:
            return await self._store_via_mcp(document, source)
        else:
            count = self.store.ingest(document, source)
            return count > 0

    async def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SemanticSearchResult]:
        """Search semantic memory for similar content."""
        if self.use_mcp:
            results = await self._search_via_mcp(query, top_k)
        else:
            results = self.store.query(query, top_k)

        return [self._format_result(r) for r in results]

    def _format_for_storage(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict]
    ) -> str:
        """Format memory entry for vector storage."""
        entry = {
            "key": key,
            "content": content,
            "metadata": metadata or {}
        }
        return f"Key: {key}\nContent: {content}\nMetadata: {json.dumps(metadata or {})}"

    def _format_result(self, raw_result: Dict) -> SemanticSearchResult:
        """Convert raw search result to SemanticSearchResult."""
        return SemanticSearchResult(
            key=raw_result.get("source", "unknown"),
            content=raw_result.get("text", ""),
            distance=raw_result.get("distance", 0.0),
            metadata=raw_result.get("metadata", {}),
            source=raw_result.get("source", "unknown")
        )
```

### Phase 2: Enhanced Similarity Search (Estimated: 1-2 days)

#### 4.2.1 Replace Keyword-Based Similarity

**File to modify:** `/ReAgent/reagent/core/memory.py`

Replace `get_similar_executions()` method (lines 204-236):

```python
async def get_similar_executions(
    self,
    task_description: str,
    top_k: int = 5,
    min_similarity: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Find similar past executions using semantic search.

    Replaces keyword-based Jaccard similarity with vector embeddings.
    """
    if not self.semantic_enabled or not self.semantic_store:
        # Fallback to original keyword-based matching
        return await self._keyword_similar_executions(task_description, top_k)

    # Use semantic search
    results = await self.semantic_store.search(
        query=task_description,
        top_k=top_k * 2  # Over-fetch for filtering
    )

    # Filter by minimum similarity and format
    similar = []
    for result in results:
        similarity = 1.0 - result.distance  # Convert distance to similarity
        if similarity >= min_similarity:
            similar.append({
                "key": result.key,
                "content": result.content,
                "similarity": similarity,
                "metadata": result.metadata
            })

    return similar[:top_k]
```

### Phase 3: Automatic Memory Indexing (Estimated: 1-2 days)

#### 4.3.1 Index Executions Automatically

**File to modify:** `/ReAgent/reagent/core/orchestrator.py`

Add semantic indexing to `_finalization_phase()` (around line 477):

```python
async def _finalization_phase(
    self,
    execution_id: str,
    result: Dict[str, Any],
    analysis: TaskComplexityAnalysis
) -> None:
    """Finalize execution and store results."""
    # Existing persistent storage
    await self.shared_memory.store_with_tier(
        key=f"execution:{execution_id}",
        value=result,
        tier=MemoryTier.PERSISTENT
    )

    # NEW: Semantic indexing for future similarity search
    if self.shared_memory.semantic_enabled:
        semantic_content = self._format_for_semantic_index(
            task=result.get("task", ""),
            analysis=analysis,
            outcome=result.get("outcome", "")
        )
        await self.shared_memory.semantic_store.store(
            key=f"execution:{execution_id}",
            content=semantic_content,
            metadata={
                "complexity": analysis.complexity_score,
                "swarm_size": analysis.recommended_swarm_size,
                "success": result.get("success", False),
                "timestamp": result.get("timestamp")
            }
        )

def _format_for_semantic_index(
    self,
    task: str,
    analysis: TaskComplexityAnalysis,
    outcome: str
) -> str:
    """Format execution data for semantic indexing."""
    return f"""
Task: {task}
Complexity: {analysis.complexity_score}/10
Domain: {', '.join(analysis.domain_assessments)}
Execution Steps: {len(analysis.execution_steps)}
Outcome: {outcome}
"""
```

### Phase 4: MCP Integration & SHARED Tier (Estimated: 2-3 days)

This phase implements the **SHARED tier** via MCP server mode, enabling multiple ReAgent instances to share semantic memory.

#### 4.4.1 Add MCP Client Implementation

**New file:** `/ReAgent/reagent/utils/mcp_client.py`

```python
"""
MCP Client for connecting to shared semantic memory server.
This enables the SHARED tier functionality for multi-agent deployments.
"""

import asyncio
import json
import subprocess
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Client for communicating with MCP servers.

    Used by SemanticMemoryAdapter in MCP mode to connect to
    a shared local-faiss-mcp server instance.
    """

    def __init__(
        self,
        server_name: str,
        config: Dict[str, Any],
        auto_start: bool = True
    ):
        self.server_name = server_name
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._connected = False

        if auto_start:
            self._start_server()

    def _start_server(self):
        """Start the MCP server subprocess."""
        command = self.config.get("command")
        args = self.config.get("args", [])

        self.process = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self._connected = True
        logger.info(f"Started MCP server: {self.server_name}")

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP tool and return the result.

        For shared semantic memory, this calls either:
        - ingest_document: Store content in shared FAISS index
        - query_rag_store: Search shared semantic memory
        """
        if not self._connected:
            raise RuntimeError("MCP client not connected")

        # MCP JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": 1
        }

        # Send request via stdio
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

        # Read response
        response_line = self.process.stdout.readline()
        response = json.loads(response_line)

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")

        return response.get("result", {})

    def close(self):
        """Shutdown the MCP server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self._connected = False
            logger.info(f"Stopped MCP server: {self.server_name}")
```

#### 4.4.2 Add MCP Server Configuration Helper

**File to modify:** `/ReAgent/reagent/utils/mcp.py`

Add semantic memory MCP support:

```python
def get_shared_semantic_memory_config(
    index_dir: str = "./reagent_shared_memory",
    embed_model: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Generate MCP configuration for shared semantic memory server.

    This configuration enables the SHARED tier - multiple ReAgent
    instances can connect to the same MCP server and share memory.
    """
    return {
        "mcpServers": {
            "shared-semantic-memory": {
                "command": "local-faiss-mcp",
                "args": [
                    "--index-dir", index_dir,
                    "--embed", embed_model
                ]
            }
        }
    }
```

#### 4.4.3 Multi-Agent Deployment Configuration

**New file:** `/ReAgent/examples/multi_agent_shared_memory.py`

```python
"""
Example: Multi-agent deployment with shared semantic memory.

This demonstrates how the SEMANTIC tier in MCP mode implements
the SHARED tier functionality.

Architecture:
    ┌─────────────────────────────────────┐
    │  local-faiss-mcp (MCP Server)       │
    │  - Single shared FAISS index        │
    │  - Persists to ./shared_memory/     │
    └─────────────┬───────────────────────┘
                  │ MCP Protocol
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
    Agent 1   Agent 2   Agent 3
    (ReAgent) (ReAgent) (ReAgent)
"""

import asyncio
from reagent import ReactiveSwarmOrchestrator, SwarmConfig

# Shared MCP configuration - all agents use the same server
SHARED_MCP_CONFIG = {
    "mcpServers": {
        "shared-semantic-memory": {
            "command": "local-faiss-mcp",
            "args": [
                "--index-dir", "./shared_semantic_memory",
                "--embed", "all-MiniLM-L6-v2"
            ]
        }
    }
}

async def create_shared_agent(agent_id: str) -> ReactiveSwarmOrchestrator:
    """Create a ReAgent instance connected to shared semantic memory."""
    return ReactiveSwarmOrchestrator(
        # Enable semantic memory in MCP (shared) mode
        semantic_memory_enabled=True,
        semantic_use_mcp=True,
        semantic_mcp_config=SHARED_MCP_CONFIG["mcpServers"]["shared-semantic-memory"],

        # Agent-specific settings
        storage_path=f"./agent_{agent_id}_local",  # Local storage still per-agent
    )

async def main():
    # Create multiple agents sharing the same semantic memory
    agents = await asyncio.gather(
        create_shared_agent("research"),
        create_shared_agent("analysis"),
        create_shared_agent("reporting")
    )

    research_agent, analysis_agent, reporting_agent = agents

    # Research agent stores findings in shared memory
    await research_agent.execute_reactive_swarm(
        task="Research market trends and store key findings",
        config=SwarmConfig(...)
    )

    # Analysis agent can now access research findings via shared semantic memory
    await analysis_agent.execute_reactive_swarm(
        task="Analyze the market trends data from research",  # Will find research results
        config=SwarmConfig(...)
    )

    # Reporting agent accesses both research and analysis
    await reporting_agent.execute_reactive_swarm(
        task="Generate report from research and analysis findings",
        config=SwarmConfig(...)
    )

if __name__ == "__main__":
    asyncio.run(main())
```

#### 4.4.2 New Reactive Tools for Semantic Memory

**File to modify:** `/ReAgent/reagent/core/orchestrator.py`

Add new tools (after line 354):

```python
@tool
def search_semantic_memory(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search semantic memory for similar past executions.

    Args:
        query: Natural language description of what to search for
        top_k: Maximum number of results to return

    Returns:
        Similar past executions with relevance scores
    """
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        self.shared_memory.get_similar_executions(query, top_k)
    )
    return {"similar_executions": results, "count": len(results)}

@tool
def store_semantic_memory(
    key: str,
    content: str,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Store content in semantic memory for future retrieval.

    Args:
        key: Unique identifier for this memory
        content: Text content to store and embed
        metadata: Optional metadata to associate with the memory

    Returns:
        Confirmation of storage
    """
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(
        self.shared_memory.semantic_store.store(key, content, metadata)
    )
    return {"success": success, "key": key}
```

### Phase 5: Configuration & Dependencies (Estimated: 0.5 days)

#### 4.5.1 Update Dependencies

**File to modify:** `/ReAgent/pyproject.toml`

Add new dependencies:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "local-faiss-mcp>=0.1.0",       # Vector store integration
    "sentence-transformers>=2.2.0",  # Embedding models
    "faiss-cpu>=1.8.0",              # Vector indexing
]

[project.optional-dependencies]
semantic = [
    "local-faiss-mcp>=0.1.0",
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.8.0",
]
```

#### 4.5.2 Update Orchestrator Constructor

**File to modify:** `/ReAgent/reagent/core/orchestrator.py`

Add semantic memory configuration (around line 81):

```python
def __init__(
    self,
    # ... existing parameters ...
    semantic_memory_enabled: bool = True,
    semantic_index_dir: str = "./reagent_semantic_memory",
    semantic_embed_model: str = "all-MiniLM-L6-v2",
    semantic_use_mcp: bool = False,
):
    # ... existing initialization ...

    # Initialize semantic memory if enabled
    semantic_config = None
    if semantic_memory_enabled:
        semantic_config = {
            "index_dir": semantic_index_dir,
            "embed_model": semantic_embed_model,
            "use_mcp": semantic_use_mcp,
        }

    self.shared_memory = ReactiveSharedMemory(
        storage_path=storage_path,
        semantic_config=semantic_config,  # NEW parameter
    )
```

---

## 5. Technical Specifications

### 5.1 Embedding Configuration

| Setting | Default | Alternatives | Notes |
|---------|---------|--------------|-------|
| Model | `all-MiniLM-L6-v2` | `all-mpnet-base-v2`, `paraphrase-multilingual-MiniLM-L12-v2` | Balance speed/quality |
| Dimension | 384 | 768 (mpnet) | Auto-detected from model |
| Chunk Size | 500 words | Configurable | With 50-word overlap |

### 5.2 Performance Characteristics

| Operation | Current (ReAgent) | With Integration | Improvement |
|-----------|-------------------|------------------|-------------|
| Similar execution search | O(n) keyword match | O(log n) vector search | 10-100x faster at scale |
| Memory retrieval accuracy | Jaccard (keywords) | Cosine (semantic) | ~26% better (per Mem0 benchmarks) |
| Storage overhead | JSON only | JSON + FAISS index | +50MB per 100k entries |

### 5.3 API Contract

#### SemanticMemoryAdapter Methods

```python
class SemanticMemoryAdapter:
    async def store(key: str, content: str, metadata: Dict) -> bool
    async def search(query: str, top_k: int) -> List[SemanticSearchResult]
    async def delete(key: str) -> bool
    async def update(key: str, content: str, metadata: Dict) -> bool
    def save() -> None  # Persist to disk
    def load() -> None  # Load from disk
```

### 5.4 Storage Format

**FAISS Index:** Binary format at `{index_dir}/faiss.index`
**Metadata:** JSON at `{index_dir}/metadata.json`

```json
{
  "model": "all-MiniLM-L6-v2",
  "documents": [
    {
      "id": 0,
      "source": "memory:execution:abc123",
      "text": "Task: Analyze market data...",
      "indexed_at": "2025-01-13T10:30:00Z"
    }
  ]
}
```

---

## 6. Alternative Approaches Considered

### 6.1 Redis for SHARED Tier (Original Approach)

**Consideration:** The original `memory.py` TODO suggested Redis for distributed storage.

**Why Rejected:**
- Adds operational complexity (Redis server management)
- Requires network infrastructure for multi-node deployment
- Overkill for most multi-agent scenarios where agents run on same machine
- Doesn't provide semantic search - would still need FAISS for that

**Why MCP Approach is Better:**
- Single solution for both SHARED + semantic search
- No external infrastructure required
- MCP server can run locally or be network-accessible
- Already aligned with ReAgent's MCP support

**When to Reconsider Redis:**
- Deployments across multiple physical machines
- Need for Redis-specific features (pub/sub, expiration)
- Enterprise environments with existing Redis infrastructure

### 6.2 Mem0 Integration

**Consideration:** Mem0 is a mature memory layer supporting multiple backends.

**Repository:** https://github.com/mem0ai/mem0

**Why Not Primary Choice:**
- Additional abstraction layer (more dependencies)
- Strands already has Mem0 integration via `strands-agents-tools`
- local_faiss_mcp provides exactly what's needed with minimal overhead

**Recommendation:** Consider Mem0 for future enhancements if:
- Graph-based memory relationships needed
- Multi-user memory scoping required
- Integration with external vector databases desired

### 6.3 Qdrant or Chroma

**Consideration:** Both are mature vector databases.

**Repositories:**
- Qdrant: https://github.com/qdrant/qdrant
- Chroma: https://github.com/chroma-core/chroma

**Why Not Primary Choice:**
- Heavier dependencies (Qdrant requires separate server)
- Chroma adds ~150MB to deployment
- local_faiss_mcp is already built for MCP integration

**Recommendation:** Consider for production deployments needing:
- Distributed vector storage
- Advanced filtering and metadata queries
- High-availability requirements

### 6.4 Separate SHARED + SEMANTIC Tiers

**Consideration:** Keep SHARED and SEMANTIC as distinct tiers.

**Why Rejected:**
- Redundant infrastructure (two systems for similar purpose)
- Complex configuration
- SHARED tier without semantic search has limited utility

**Why Unified Approach is Better:**
- One system, two modes (direct vs MCP)
- Simpler mental model
- SEMANTIC automatically enables sharing when in MCP mode

---

## 7. References

### 7.1 Repository Locations (Local)

| Repository | Local Path |
|------------|------------|
| ReAgent | `/Users/fnp/tmp/ReAgent/` |
| local_faiss_mcp | `/Users/fnp/tmp/local_faiss_mcp/` |

### 7.2 Key Files to Modify

| File | Lines | Changes |
|------|-------|---------|
| `ReAgent/reagent/core/memory.py` | 42, 65, 204-236, 376, 546 | Replace SHARED with SEMANTIC tier, enhance similarity |
| `ReAgent/reagent/core/orchestrator.py` | 81-149, 354, 477 | Add config, tools, indexing |
| `ReAgent/reagent/utils/mcp.py` | - | Add shared memory MCP config helper |
| `ReAgent/pyproject.toml` | dependencies | Add local-faiss-mcp |
| NEW: `ReAgent/reagent/core/semantic_memory.py` | - | SemanticMemoryAdapter class (direct + MCP modes) |
| NEW: `ReAgent/reagent/utils/mcp_client.py` | - | MCPClient for SHARED tier via MCP |
| NEW: `ReAgent/examples/multi_agent_shared_memory.py` | - | Multi-agent SHARED tier example |

### 7.3 External Documentation

| Resource | URL |
|----------|-----|
| Strands Agents SDK | https://github.com/strands-agents/sdk-python |
| Strands Documentation | https://strandsagents.com/latest/ |
| FAISS Documentation | https://github.com/facebookresearch/faiss/wiki |
| Sentence Transformers | https://www.sbert.net/ |
| MCP Protocol | https://modelcontextprotocol.io |
| Mem0 (Reference) | https://github.com/mem0ai/mem0 |
| LangGraph Memory Docs | https://langchain-ai.github.io/langgraph/concepts/memory/ |

### 7.4 Related Open Source Projects

| Project | URL | Relevance |
|---------|-----|-----------|
| Letta (MemGPT) | https://github.com/letta-ai/letta | Advanced memory architecture patterns |
| CrewAI | https://github.com/crewAIInc/crewAI | Multi-agent memory sharing |
| LlamaIndex | https://github.com/run-llama/llama_index | Vector store integrations |

---

## 8. Success Criteria

### 8.1 Functional Requirements

- [ ] Semantic search returns relevant results for task descriptions
- [ ] Past executions are automatically indexed in vector store
- [ ] MCP integration works with Claude Code and other MCP clients
- [ ] Backward compatibility with existing ReAgent installations
- [ ] Graceful fallback when semantic memory is disabled
- [ ] **SHARED tier: Multiple ReAgent instances can share semantic memory via MCP**
- [ ] **SHARED tier: Memory persists when individual agents restart**
- [ ] **SHARED tier: Agent A's stored memories are searchable by Agent B**

### 8.2 Performance Requirements

- [ ] Semantic search < 50ms for 10k indexed entries
- [ ] Memory storage overhead < 100MB for 50k entries
- [ ] No impact on execution path when semantic memory disabled
- [ ] Index persistence survives process restarts
- [ ] **MCP mode: < 20ms overhead per operation vs direct mode**

### 8.3 Quality Requirements

- [ ] Unit tests for SemanticMemoryAdapter
- [ ] Integration tests with local_faiss_mcp
- [ ] Documentation updated in README
- [ ] Type hints for all new public APIs
- [ ] **Multi-agent example demonstrating SHARED tier**
- [ ] **MCP client tests for shared memory operations**

### 8.4 SHARED Tier Specific Criteria

- [ ] Single MCP server serves multiple ReAgent instances
- [ ] No data loss when agents connect/disconnect
- [ ] Concurrent read/write operations handled correctly
- [ ] Clear documentation on when to use MCP vs direct mode

---

## Appendix A: Quick Start Guide

### Installation

```bash
# Install ReAgent with semantic memory support
pip install reagent[semantic]

# Or install dependencies manually
pip install local-faiss-mcp sentence-transformers faiss-cpu
```

### Single-Agent Mode (Direct Library)

Best for: Single agent deployments, maximum performance

```python
from reagent import ReactiveSwarmOrchestrator, SwarmConfig

# Enable semantic memory in direct mode (no sharing)
orchestrator = ReactiveSwarmOrchestrator(
    semantic_memory_enabled=True,
    semantic_use_mcp=False,              # Direct library mode
    semantic_index_dir="./my_agent_memory",
    semantic_embed_model="all-MiniLM-L6-v2"
)

# Execute task (automatically indexed for future reference)
result = await orchestrator.execute_reactive_swarm(
    task="Analyze quarterly sales data and identify trends",
    config=SwarmConfig(...)
)

# Search past executions using semantic similarity
similar = await orchestrator.shared_memory.get_similar_executions(
    "Find patterns in revenue data",
    top_k=5
)
```

### Multi-Agent Mode (MCP Server = SHARED Tier)

Best for: Multiple agents that need to share memory

**Step 1: Start the shared MCP server (one instance for all agents)**

```bash
# Run in background or separate terminal
local-faiss-mcp --index-dir ./shared_semantic_memory --embed all-MiniLM-L6-v2
```

**Step 2: Configure agents to connect to shared server**

```python
from reagent import ReactiveSwarmOrchestrator, SwarmConfig

# MCP configuration for shared memory
SHARED_MCP_CONFIG = {
    "command": "local-faiss-mcp",
    "args": ["--index-dir", "./shared_semantic_memory", "--embed", "all-MiniLM-L6-v2"]
}

# Create multiple agents sharing the same semantic memory
research_agent = ReactiveSwarmOrchestrator(
    semantic_memory_enabled=True,
    semantic_use_mcp=True,               # MCP mode = SHARED tier
    semantic_mcp_config=SHARED_MCP_CONFIG,
    storage_path="./research_agent_local"
)

analysis_agent = ReactiveSwarmOrchestrator(
    semantic_memory_enabled=True,
    semantic_use_mcp=True,               # Same shared memory
    semantic_mcp_config=SHARED_MCP_CONFIG,
    storage_path="./analysis_agent_local"
)

# Research agent stores findings
await research_agent.execute_reactive_swarm(
    task="Research market trends for Q4",
    config=SwarmConfig(...)
)

# Analysis agent can find research agent's memories!
similar = await analysis_agent.shared_memory.get_similar_executions(
    "What do we know about market trends?",
    top_k=5
)
# Returns research agent's Q4 findings
```

### MCP Configuration File

For use with Claude Code or other MCP clients:

```json
{
  "mcpServers": {
    "shared-semantic-memory": {
      "command": "local-faiss-mcp",
      "args": [
        "--index-dir", "./reagent_shared_memory",
        "--embed", "all-MiniLM-L6-v2"
      ]
    }
  }
}
```

### When to Use Each Mode

| Scenario | Mode | Configuration |
|----------|------|---------------|
| Single agent, best performance | Direct | `semantic_use_mcp=False` |
| Multiple agents, same machine | MCP | `semantic_use_mcp=True` |
| Agents across machines | MCP + network | MCP server on shared host |
| Testing/development | Direct | Simpler setup |
| Production multi-agent | MCP | Memory persists independently |

---

*Document generated: 2025-01-13*
*Last updated: 2025-01-14*
*Author: AI Research Agent*
*Version: 1.1*

**Changelog:**
- v1.1: Revised to unify SEMANTIC and SHARED tiers via MCP server mode
- v1.0: Initial plan with separate SEMANTIC tier proposal
