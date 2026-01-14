# Evaluation Driven Development (EDD) Test Plan
## ReAgent Semantic Shared Memory Integration

---

## Document Purpose

This document follows the **Evaluation Driven Development (EDD)** paradigm - a test-driven approach adapted for stochastic LLM applications. Unlike traditional TDD where deterministic tests are written first, EDD focuses on:

1. **Problem Space Definition** - What problem are we actually solving?
2. **Evaluation Dataset Design** - Golden datasets that capture real-world scenarios
3. **Evaluation Methodology** - Metrics that account for LLM non-determinism
4. **Validation Test Design** - Tests that assess solution fitness
5. **Solution Fitness Assessment** - Is this the right tool for the problem?

---

## Table of Contents

1. [Problem Space Analysis](#1-problem-space-analysis)
2. [Evaluation Dimensions](#2-evaluation-dimensions)
3. [Golden Datasets](#3-golden-datasets)
4. [Evaluation Methodology](#4-evaluation-methodology)
5. [Validation Test Design](#5-validation-test-design)
6. [Solution Fitness Assessment](#6-solution-fitness-assessment)
7. [Baseline Comparisons](#7-baseline-comparisons)
8. [Implementation Verification Tests](#8-implementation-verification-tests)

---

## 1. Problem Space Analysis

### 1.1 Core Problem Statement

> **When an AI agent executes a task, how effectively can it leverage knowledge from past executions to improve future performance?**

This breaks down into sub-problems:

| Sub-Problem | Description | Current State (ReAgent) |
|-------------|-------------|------------------------|
| **P1: Memory Retrieval Accuracy** | Finding relevant past executions given a new task | Keyword-based Jaccard similarity (~60% accuracy) |
| **P2: Cross-Domain Transfer** | Applying learnings from one domain to another | Not supported (keywords don't transfer) |
| **P3: Multi-Agent Knowledge Sharing** | Agent A's learnings available to Agent B | SHARED tier unimplemented |
| **P4: Scalable Search** | Performance at scale (10k+ memories) | O(n) linear search |
| **P5: Context Preservation** | Rich context survives storage/retrieval | Partial (metadata stored, not semantically indexed) |

### 1.2 Problem Space Boundaries

**In Scope:**
- Task-to-task semantic similarity
- Execution outcome retrieval
- Multi-agent memory sharing
- Performance at realistic scales (1k-100k memories)

**Out of Scope:**
- Real-time collaboration (agents modifying same memory simultaneously)
- Memory versioning/history
- Access control/permissions
- Cross-model memory (different LLM providers)

### 1.3 Stakeholder Scenarios

#### Scenario S1: Single Agent Learning
```
Agent executes: "Analyze Q3 sales data for North America"
Agent later receives: "Review revenue trends for US market"
Expected: Agent finds Q3 analysis as relevant prior execution
```

#### Scenario S2: Cross-Domain Transfer
```
Agent executes: "Debug timeout errors in payment service"
Agent later receives: "Fix connection issues in notification service"
Expected: Agent finds timeout debugging as conceptually similar
```

#### Scenario S3: Multi-Agent Handoff
```
Research Agent completes: "Investigate competitor pricing strategies"
Analysis Agent receives: "Compare our pricing to market"
Expected: Analysis Agent finds Research Agent's findings
```

#### Scenario S4: Negative Case - Irrelevant Retrieval
```
Agent executes: "Generate monthly expense report"
Agent later receives: "Write unit tests for authentication"
Expected: Agent does NOT retrieve expense report as relevant
```

---

## 2. Evaluation Dimensions

### 2.1 Primary Dimensions

| Dimension | Definition | Why It Matters |
|-----------|------------|----------------|
| **Semantic Accuracy** | Relevant memories retrieved for semantically similar queries | Core value proposition |
| **Precision@K** | Of top-K results, what % are truly relevant | Noise in results wastes LLM context |
| **Recall** | Of all relevant memories, what % are retrieved | Missing relevant context hurts outcomes |
| **Cross-Agent Consistency** | Same query returns same results across agents | SHARED tier correctness |
| **Latency** | Time to store/retrieve | Must not bottleneck agent execution |
| **Scalability** | Performance degradation as memory grows | Production viability |

### 2.2 Stochastic Considerations

LLM-based systems introduce non-determinism at multiple levels:

| Source of Variance | Impact | Mitigation in Evaluation |
|-------------------|--------|--------------------------|
| **Embedding Model** | Same text → slightly different vectors | Use fixed model version, measure variance |
| **Query Formulation** | LLM may phrase same intent differently | Test with paraphrased queries |
| **Relevance Judgment** | Human/LLM judges may disagree | Multiple judges, inter-rater reliability |
| **Temperature Effects** | Non-zero temp affects downstream usage | Evaluate retrieval independent of generation |

### 2.3 Evaluation Metrics

```python
# Primary Metrics
semantic_accuracy = relevant_retrieved / total_retrieved  # Precision
recall_at_k = relevant_retrieved / total_relevant         # Recall
f1_score = 2 * (precision * recall) / (precision + recall)

# Ranking Quality
mrr = mean(1 / rank_of_first_relevant)  # Mean Reciprocal Rank
ndcg = normalized_discounted_cumulative_gain(rankings)

# Stochastic Stability
variance_coefficient = std(scores_across_runs) / mean(scores_across_runs)
consistency_rate = same_top_k_across_n_runs / n_runs

# Performance
p50_latency = percentile(latencies, 50)
p99_latency = percentile(latencies, 99)
throughput = queries_per_second
```

---

## 3. Golden Datasets

### 3.1 Dataset Design Principles

1. **Representative** - Cover realistic agent task distributions
2. **Labeled** - Human-verified relevance judgments
3. **Diverse** - Multiple domains, complexity levels
4. **Balanced** - Include negative cases (irrelevant pairs)
5. **Reproducible** - Fixed dataset for consistent evaluation

### 3.2 Sourcing Strategy

Based on open-source community research, we can source significant portions of our datasets:

| Dataset | Source % | Generate % | Primary Sources |
|---------|----------|------------|-----------------|
| TSB | 75% | 25% | STSB, Super-NaturalInstructions, FLAN |
| CDTB | 80% | 20% | BEIR, BIG-Bench, MS MARCO |
| MAHB | 50% | 50% | CAMEL, AgentBench *(gap in community)* |
| NCB | 75% | 25% | All-NLI, HH-RLHF, HelpSteer |
| SST | 85% | 15% | MTEB, BEIR, MS MARCO |

**Key Open-Source Datasets:**

| Dataset | URL | Size | Use Case |
|---------|-----|------|----------|
| STSB | huggingface.co/datasets/sentence-transformers/stsb | 8.6K pairs | Baseline similarity scores (0-1) |
| All-NLI | huggingface.co/datasets/sentence-transformers/all-nli | 2.86M rows | Hard negatives (contradiction pairs) |
| Super-NaturalInstructions | arxiv.org/abs/2204.07705 | 1,616 tasks | Task instruction similarity |
| BEIR | github.com/beir-cellar/beir | 17+ datasets | Cross-domain retrieval |
| CAMEL | github.com/camel-ai/camel | Multi-agent convos | Handoff pattern extraction |
| MTEB | huggingface.co/spaces/mteb/leaderboard | Comprehensive | Embedding quality baseline |

**Critical Gap:** No standardized multi-agent handoff benchmark exists. MAHB requires 50% custom generation and is a candidate for open-source contribution.

### 3.3 Dataset 1: Task Similarity Benchmark (TSB)

**Purpose:** Evaluate semantic similarity between agent tasks

**Sourcing Strategy (75% sourced, 25% generated):**
```
TSB Dataset Composition:
├── STSB baseline pairs (20%)           ← Direct from sentence-transformers/stsb
├── Super-NaturalInstructions (35%)     ← Derived: create pairs from 1,616 task definitions
├── FLAN template variations (20%)      ← Derived: same-task templates = high similarity
└── Custom agent task pairs (25%)       ← Generated: ReAgent-specific scenarios
```

**Structure:**
```json
{
  "dataset": "task_similarity_benchmark",
  "version": "1.0",
  "source_breakdown": {
    "stsb": 0.20,
    "super_natural_instructions": 0.35,
    "flan_templates": 0.20,
    "custom_generated": 0.25
  },
  "entries": [
    {
      "id": "tsb_001",
      "source": "custom_generated",
      "query_task": "Analyze customer churn patterns in Q4 subscription data",
      "candidate_tasks": [
        {
          "task": "Review subscriber retention metrics for last quarter",
          "relevance": 0.9,
          "relevance_rationale": "Same domain (subscriptions), same time frame, related goal"
        },
        {
          "task": "Investigate user drop-off in mobile app onboarding",
          "relevance": 0.6,
          "relevance_rationale": "Related concept (drop-off/churn) but different context"
        },
        {
          "task": "Generate Q4 financial summary report",
          "relevance": 0.3,
          "relevance_rationale": "Same time frame but different focus"
        },
        {
          "task": "Update employee vacation calendar",
          "relevance": 0.0,
          "relevance_rationale": "Completely unrelated"
        }
      ]
    }
  ]
}
```

**Dataset Size:** 200 query tasks, 800 candidate tasks (4 per query)

**Relevance Scale:**
- 0.0 - Irrelevant
- 0.3 - Tangentially related
- 0.6 - Conceptually similar
- 0.9 - Highly relevant
- 1.0 - Near-duplicate

**Sourcing Scripts:**
```python
# scripts/source_tsb_from_stsb.py
from datasets import load_dataset

def source_stsb_pairs():
    """Load STSB and convert to TSB format."""
    stsb = load_dataset("sentence-transformers/stsb", split="test")

    tsb_entries = []
    for row in stsb:
        tsb_entries.append({
            "id": f"tsb_stsb_{row['idx']}",
            "source": "stsb",
            "query_task": row["sentence1"],
            "candidate_tasks": [{
                "task": row["sentence2"],
                "relevance": row["score"],  # Already 0-1 normalized
                "relevance_rationale": "STSB human annotation"
            }]
        })
    return tsb_entries

# scripts/source_tsb_from_supernatural.py
def source_supernatural_pairs():
    """Create task pairs from Super-NaturalInstructions based on task type."""
    # Tasks with same type = high similarity
    # Tasks with different type = low similarity
    pass
```

### 3.4 Dataset 2: Cross-Domain Transfer Benchmark (CDTB)

**Purpose:** Evaluate ability to find conceptually similar tasks across different domains

**Sourcing Strategy (80% sourced, 20% generated):**
```
CDTB Dataset Composition:
├── BEIR cross-domain queries (40%)     ← Direct from 17+ domain datasets
├── BIG-Bench cross-task pairs (25%)    ← Derived: 167 task subsets across domains
├── MS MARCO multi-domain (15%)         ← Derived: real Bing queries span domains
└── Custom agent scenarios (20%)        ← Generated: ReAgent-specific cross-domain
```

**Key BEIR Domains to Use:**
- TREC-COVID (medical)
- FiQA (finance)
- SciFact (science)
- NFCorpus (nutrition)
- ArguAna (argumentation)
- CQADupstack (12 StackExchange domains)

**Structure:**
```json
{
  "dataset": "cross_domain_transfer_benchmark",
  "version": "1.0",
  "source_breakdown": {
    "beir": 0.40,
    "bigbench": 0.25,
    "msmarco": 0.15,
    "custom_generated": 0.20
  },
  "entries": [
    {
      "id": "cdtb_001",
      "source": "custom_generated",
      "query_task": "Debug memory leak in Java payment service",
      "query_domain": "backend_engineering",
      "candidate_tasks": [
        {
          "task": "Fix resource exhaustion in Python data pipeline",
          "domain": "data_engineering",
          "transfer_relevance": 0.8,
          "transfer_rationale": "Same problem pattern (resource leak) different tech stack"
        },
        {
          "task": "Optimize database query performance",
          "domain": "database",
          "transfer_relevance": 0.4,
          "transfer_rationale": "Performance optimization but different root cause"
        }
      ]
    }
  ]
}
```

**Dataset Size:** 100 query tasks across 10 domains

**Domains:**
1. Backend Engineering
2. Frontend Development
3. Data Engineering
4. DevOps/Infrastructure
5. Security
6. Database
7. ML/AI
8. Mobile Development
9. API Design
10. Testing/QA

**Sourcing Scripts:**
```python
# scripts/source_cdtb_from_beir.py
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def source_beir_cross_domain():
    """Extract cross-domain queries from BEIR datasets."""
    domains = ["trec-covid", "fiqa", "scifact", "nfcorpus", "arguana"]

    cdtb_entries = []
    for domain in domains:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{domain}.zip"
        data_path = util.download_and_unzip(url, "datasets")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        for qid, query in queries.items():
            cdtb_entries.append({
                "id": f"cdtb_beir_{domain}_{qid}",
                "source": "beir",
                "query_task": query,
                "query_domain": domain,
                # Match against other domains for cross-domain pairs
            })
    return cdtb_entries
```

### 3.5 Dataset 3: Multi-Agent Handoff Benchmark (MAHB)

**Purpose:** Evaluate SHARED tier - memories stored by Agent A retrievable by Agent B

**⚠️ CRITICAL GAP: No standardized benchmark exists for multi-agent memory handoff.**

This is a significant gap in the open-source community. MAHB requires substantial custom generation and is a candidate for open-source contribution.

**Sourcing Strategy (50% sourced, 50% generated):**
```
MAHB Dataset Composition:
├── CAMEL role-playing extracts (25%)   ← Derived: extract handoff patterns from conversations
├── AgentBench adaptation (15%)         ← Derived: multi-turn interactions as handoff scenarios
├── ChatDev/AutoGen patterns (10%)      ← Inspired: use collaboration patterns as templates
└── Custom generation (50%)             ← Generated: ReAgent-specific handoff scenarios
```

**Why This Gap Exists:**
- Multi-agent systems are nascent (2023-2024 emergence)
- Most benchmarks focus on single-agent capabilities
- Handoff is implicit in existing frameworks, not measured

**Community Contribution Opportunity:**
Consider publishing MAHB as open-source benchmark after validation.

**Structure:**
```json
{
  "dataset": "multi_agent_handoff_benchmark",
  "version": "1.0",
  "community_contribution": true,
  "source_breakdown": {
    "camel_derived": 0.25,
    "agentbench_adapted": 0.15,
    "pattern_inspired": 0.10,
    "custom_generated": 0.50
  },
  "entries": [
    {
      "id": "mahb_001",
      "source": "custom_generated",
      "storing_agent": "research_agent",
      "stored_execution": {
        "task": "Research competitor pricing for cloud storage services",
        "outcome": "Found 3 competitors: AWS S3, Azure Blob, GCP Storage. Pricing analysis completed.",
        "metadata": {
          "domain": "competitive_analysis",
          "timestamp": "2025-01-10T10:00:00Z"
        }
      },
      "retrieving_agent": "strategy_agent",
      "retrieval_query": "What do we know about cloud storage market pricing?",
      "expected_retrieval": true,
      "min_similarity_score": 0.7
    }
  ]
}
```

**Dataset Size:** 150 handoff scenarios across 5 agent role combinations

**Agent Roles:**
- Research Agent → Analysis Agent
- Analysis Agent → Reporting Agent
- Development Agent → Testing Agent
- Planning Agent → Execution Agent
- Monitoring Agent → Remediation Agent

**Sourcing Scripts:**
```python
# scripts/source_mahb_from_camel.py
from datasets import load_dataset

def extract_camel_handoffs():
    """Extract handoff patterns from CAMEL role-playing conversations."""
    # CAMEL datasets: AI Society, Code, Math, Physics, Chemistry, Biology
    camel_data = load_dataset("camel-ai/ai_society", split="train")

    mahb_entries = []
    for conv in camel_data:
        # Look for knowledge transfer moments in conversation
        # Where one agent shares findings that another could use
        messages = conv["messages"]
        for i, msg in enumerate(messages):
            if contains_knowledge_transfer(msg):
                mahb_entries.append({
                    "id": f"mahb_camel_{conv['id']}_{i}",
                    "source": "camel_derived",
                    "storing_agent": msg["role"],
                    "stored_execution": extract_knowledge(msg),
                    "retrieving_agent": get_other_role(msg["role"]),
                    "retrieval_query": generate_retrieval_query(msg),
                })
    return mahb_entries

def contains_knowledge_transfer(msg):
    """Detect if message contains shareable knowledge."""
    indicators = ["found that", "discovered", "analysis shows", "results indicate"]
    return any(ind in msg["content"].lower() for ind in indicators)
```

### 3.6 Dataset 4: Negative Cases Benchmark (NCB)

**Purpose:** Ensure irrelevant memories are NOT retrieved (precision)

**Sourcing Strategy (75% sourced, 25% generated):**
```
NCB Dataset Composition:
├── All-NLI contradiction pairs (35%)   ← Direct: 571K triplets with hard negatives
├── HH-RLHF rejected responses (20%)    ← Derived: rejected = irrelevant
├── HelpSteer low-score samples (15%)   ← Derived: low relevance scores
├── BEIR non-relevant passages (5%)     ← Direct: passages not matching query
└── Custom irrelevant scenarios (25%)   ← Generated: agent-specific negatives
```

**Key Source: All-NLI (Contradiction Pairs)**
- 571K triplets in format (anchor, positive, negative)
- Contradiction pairs provide linguistically hard negatives
- Human-annotated via NLI task

**Structure:**
```json
{
  "dataset": "negative_cases_benchmark",
  "version": "1.0",
  "source_breakdown": {
    "all_nli_contradictions": 0.35,
    "hh_rlhf_rejected": 0.20,
    "helpsteer_low_score": 0.15,
    "beir_non_relevant": 0.05,
    "custom_generated": 0.25
  },
  "entries": [
    {
      "id": "ncb_001",
      "source": "custom_generated",
      "query_task": "Write Python unit tests for user authentication",
      "irrelevant_memories": [
        {
          "task": "Plan team offsite agenda for Q2",
          "should_retrieve": false,
          "max_allowed_similarity": 0.2
        },
        {
          "task": "Review marketing campaign metrics",
          "should_retrieve": false,
          "max_allowed_similarity": 0.2
        }
      ]
    }
  ]
}
```

**Dataset Size:** 100 queries with 500 confirmed irrelevant candidates

**Sourcing Scripts:**
```python
# scripts/source_ncb_from_allnli.py
from datasets import load_dataset

def source_allnli_negatives():
    """Extract hard negatives from All-NLI contradiction pairs."""
    allnli = load_dataset("sentence-transformers/all-nli", "triplet", split="train")

    ncb_entries = []
    for row in allnli:
        ncb_entries.append({
            "id": f"ncb_allnli_{row['idx']}",
            "source": "all_nli_contradictions",
            "query_task": row["anchor"],
            "irrelevant_memories": [{
                "task": row["negative"],  # Contradiction = hard negative
                "should_retrieve": False,
                "max_allowed_similarity": 0.3,
                "negative_type": "contradiction"
            }]
        })
    return ncb_entries

# scripts/source_ncb_from_hhrlhf.py
def source_hhrlhf_negatives():
    """Use rejected responses from HH-RLHF as irrelevant examples."""
    hhrlhf = load_dataset("Anthropic/hh-rlhf", split="train")

    ncb_entries = []
    for row in hhrlhf:
        # Extract the rejected response as a negative example
        ncb_entries.append({
            "id": f"ncb_hhrlhf_{hash(row['chosen'][:50])}",
            "source": "hh_rlhf_rejected",
            "query_task": extract_query_from_chosen(row["chosen"]),
            "irrelevant_memories": [{
                "task": extract_task_from_rejected(row["rejected"]),
                "should_retrieve": False,
                "max_allowed_similarity": 0.25
            }]
        })
    return ncb_entries
```

### 3.7 Dataset 5: Scale Stress Test (SST)

**Purpose:** Evaluate performance degradation at scale

**Sourcing Strategy (85% sourced, 15% generated):**
```
SST Dataset Composition:
├── MS MARCO passages (45%)             ← Direct: 8.8M passages for scale testing
├── BEIR corpora (25%)                  ← Direct: various corpus sizes
├── MTEB retrieval datasets (15%)       ← Direct: standardized scale benchmarks
└── Synthetic agent memories (15%)      ← Generated: ReAgent-specific format
```

**Why MS MARCO for Scale:**
- 8.8M passages available
- Real-world distribution of content
- Industry standard for retrieval benchmarking
- Pre-computed embeddings available

**Structure:**
- 1,000 memories (baseline)
- 10,000 memories (medium scale)
- 50,000 memories (large scale)
- 100,000 memories (stress test)

**Metrics per scale:**
- p50, p95, p99 latency
- Throughput (queries/second)
- Memory usage
- Index size on disk

**Sourcing Scripts:**
```python
# scripts/source_sst_from_msmarco.py
from datasets import load_dataset

def source_msmarco_for_scale(target_size: int):
    """Load MS MARCO passages for scale testing."""
    msmarco = load_dataset("ms_marco", "v2.1", split="train")

    # Sample to target size
    sampled = msmarco.shuffle(seed=42).select(range(min(target_size, len(msmarco))))

    sst_entries = []
    for idx, row in enumerate(sampled):
        sst_entries.append({
            "id": f"sst_msmarco_{idx}",
            "source": "msmarco",
            "content": row["passages"]["passage_text"][0] if row["passages"]["passage_text"] else "",
            "metadata": {
                "query": row["query"],
                "is_selected": row["passages"]["is_selected"][0] if row["passages"]["is_selected"] else False
            }
        })
    return sst_entries
```

---

## 4. Evaluation Methodology

### 4.1 Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EDD Evaluation Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Golden    │    │   System    │    │  Retrieval  │    │   Metric    │  │
│  │  Dataset    │───►│   Under     │───►│   Results   │───►│ Computation │  │
│  │   (Input)   │    │    Test     │    │             │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘  │
│                                                                   │         │
│                     ┌─────────────────────────────────────────────┘         │
│                     │                                                        │
│                     ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      Evaluation Report                                   ││
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            ││
│  │  │ Precision │  │  Recall   │  │    MRR    │  │  Latency  │            ││
│  │  │   @K      │  │    @K     │  │           │  │   P99     │            ││
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Evaluation Harness

```python
"""
EDD Evaluation Harness for ReAgent Semantic Memory
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import time
import statistics

@dataclass
class EvaluationResult:
    """Result from a single evaluation query."""
    query_id: str
    query_task: str
    retrieved_memories: List[Dict[str, Any]]
    retrieval_latency_ms: float
    expected_relevant: List[str]

@dataclass
class EvaluationReport:
    """Aggregated evaluation metrics."""
    dataset_name: str
    system_name: str
    timestamp: str

    # Accuracy Metrics
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_5: float
    recall_at_10: float
    f1_at_5: float
    mrr: float
    ndcg_at_10: float

    # Stochastic Stability
    variance_coefficient: float
    consistency_rate: float

    # Performance Metrics
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_qps: float

    # Scale Metrics (if applicable)
    memory_count: int
    index_size_mb: float

class EDDEvaluator:
    """
    Evaluation Driven Development harness for semantic memory.

    Runs golden datasets against system under test and computes metrics.
    """

    def __init__(
        self,
        system_under_test: "SemanticMemorySystem",
        golden_datasets_dir: str
    ):
        self.sut = system_under_test
        self.datasets_dir = golden_datasets_dir

    async def evaluate_dataset(
        self,
        dataset_name: str,
        num_runs: int = 3  # For stochastic stability measurement
    ) -> EvaluationReport:
        """
        Run evaluation on a golden dataset.

        Args:
            dataset_name: Name of dataset (e.g., "task_similarity_benchmark")
            num_runs: Number of runs for measuring variance

        Returns:
            EvaluationReport with all metrics
        """
        dataset = self._load_dataset(dataset_name)

        all_results = []
        for run_idx in range(num_runs):
            run_results = await self._execute_evaluation_run(dataset)
            all_results.append(run_results)

        return self._compute_metrics(dataset_name, all_results)

    async def _execute_evaluation_run(
        self,
        dataset: Dict
    ) -> List[EvaluationResult]:
        """Execute single evaluation run."""
        results = []

        for entry in dataset["entries"]:
            start_time = time.perf_counter()

            retrieved = await self.sut.search(
                query=entry["query_task"],
                top_k=10
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            results.append(EvaluationResult(
                query_id=entry["id"],
                query_task=entry["query_task"],
                retrieved_memories=retrieved,
                retrieval_latency_ms=latency_ms,
                expected_relevant=self._get_relevant_ids(entry)
            ))

        return results

    def _compute_precision_at_k(
        self,
        results: List[EvaluationResult],
        k: int
    ) -> float:
        """Compute Precision@K across all results."""
        precisions = []

        for result in results:
            top_k = result.retrieved_memories[:k]
            relevant_in_top_k = sum(
                1 for m in top_k
                if m["id"] in result.expected_relevant
            )
            precisions.append(relevant_in_top_k / k if k > 0 else 0)

        return statistics.mean(precisions)

    def _compute_mrr(self, results: List[EvaluationResult]) -> float:
        """Compute Mean Reciprocal Rank."""
        reciprocal_ranks = []

        for result in results:
            for rank, memory in enumerate(result.retrieved_memories, 1):
                if memory["id"] in result.expected_relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return statistics.mean(reciprocal_ranks)

    def _compute_variance_coefficient(
        self,
        all_runs: List[List[EvaluationResult]]
    ) -> float:
        """Measure stochastic stability across runs."""
        # Compare top-5 results across runs for each query
        # Lower variance = more stable
        pass  # Implementation details
```

### 4.3 Relevance Judgment Protocol

Since relevance is subjective, we use multiple judgment sources:

#### 4.3.1 Human Judges
- 3 judges per query-candidate pair
- Inter-rater reliability measured via Fleiss' Kappa
- Disagreements resolved by majority vote

#### 4.3.2 LLM-as-Judge (Secondary)
```python
RELEVANCE_JUDGMENT_PROMPT = """
You are evaluating whether a past agent execution is relevant to a new task.

New Task: {query_task}

Past Execution:
- Task: {candidate_task}
- Outcome: {candidate_outcome}

Rate the relevance on a scale of 0.0 to 1.0:
- 0.0: Completely irrelevant
- 0.3: Tangentially related
- 0.6: Conceptually similar, may provide useful context
- 0.9: Highly relevant, directly applicable
- 1.0: Near-duplicate task

Provide your rating and a brief rationale.

Rating:
Rationale:
"""
```

#### 4.3.3 Programmatic Rules (Tertiary)
- Keyword overlap threshold
- Domain classification match
- Time proximity (for time-sensitive tasks)

### 4.4 Statistical Significance

For comparing systems (baseline vs new), we require:

- **Minimum sample size:** 100 queries per dataset
- **Significance level:** α = 0.05
- **Statistical test:** Paired t-test for metric comparisons
- **Effect size:** Report Cohen's d alongside p-values

---

## 5. Validation Test Design

### 5.1 Test Categories

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EDD Test Pyramid                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              ┌───────────┐                                   │
│                              │  E2E      │  Multi-agent scenarios            │
│                              │  Tests    │  (MAHB dataset)                   │
│                              └─────┬─────┘                                   │
│                            ┌───────┴───────┐                                 │
│                            │  Integration  │  Cross-system tests             │
│                            │    Tests      │  (CDTB dataset)                 │
│                            └───────┬───────┘                                 │
│                      ┌─────────────┴─────────────┐                           │
│                      │      Evaluation Tests      │  Golden dataset          │
│                      │       (Core EDD)           │  validation              │
│                      └─────────────┬─────────────┘                           │
│              ┌─────────────────────┴─────────────────────┐                   │
│              │            Unit Tests                      │  Deterministic    │
│              │     (Traditional TDD for components)       │  component tests  │
│              └────────────────────────────────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Evaluation Tests (Core EDD)

These tests validate against golden datasets with statistical thresholds:

```python
import pytest
from edd_evaluator import EDDEvaluator, SemanticMemorySystem

class TestSemanticMemoryEvaluation:
    """
    EDD Evaluation Tests - Run against golden datasets.

    These tests validate the QUALITY of the semantic memory system,
    not just its functionality.
    """

    @pytest.fixture
    def evaluator(self):
        system = SemanticMemorySystem(
            use_mcp=False,  # Direct mode for evaluation
            index_dir="./test_index"
        )
        return EDDEvaluator(system, "./golden_datasets")

    # ============================================================
    # Task Similarity Benchmark (TSB) Tests
    # ============================================================

    @pytest.mark.edd
    async def test_tsb_precision_at_5_exceeds_baseline(self, evaluator):
        """
        REQUIREMENT: Precision@5 must exceed keyword baseline by 20%+

        Baseline (Jaccard keyword): ~0.55
        Target (Semantic): >= 0.66
        """
        report = await evaluator.evaluate_dataset("task_similarity_benchmark")

        BASELINE_PRECISION = 0.55
        MINIMUM_IMPROVEMENT = 0.20
        TARGET = BASELINE_PRECISION * (1 + MINIMUM_IMPROVEMENT)

        assert report.precision_at_5 >= TARGET, (
            f"Precision@5 ({report.precision_at_5:.3f}) below target ({TARGET:.3f}). "
            f"Semantic memory must outperform keyword baseline by 20%+"
        )

    @pytest.mark.edd
    async def test_tsb_recall_at_10_minimum(self, evaluator):
        """
        REQUIREMENT: Recall@10 must be >= 0.80

        We cannot miss more than 20% of relevant memories in top 10.
        """
        report = await evaluator.evaluate_dataset("task_similarity_benchmark")

        MINIMUM_RECALL = 0.80

        assert report.recall_at_10 >= MINIMUM_RECALL, (
            f"Recall@10 ({report.recall_at_10:.3f}) below minimum ({MINIMUM_RECALL}). "
            f"Too many relevant memories are being missed."
        )

    @pytest.mark.edd
    async def test_tsb_mrr_minimum(self, evaluator):
        """
        REQUIREMENT: MRR must be >= 0.70

        First relevant result should typically appear in top 2.
        """
        report = await evaluator.evaluate_dataset("task_similarity_benchmark")

        MINIMUM_MRR = 0.70

        assert report.mrr >= MINIMUM_MRR, (
            f"MRR ({report.mrr:.3f}) below minimum ({MINIMUM_MRR}). "
            f"Relevant memories not ranking high enough."
        )

    # ============================================================
    # Cross-Domain Transfer Benchmark (CDTB) Tests
    # ============================================================

    @pytest.mark.edd
    async def test_cdtb_cross_domain_recall(self, evaluator):
        """
        REQUIREMENT: Cross-domain recall@5 must be >= 0.60

        System must find conceptually similar tasks across different domains.
        This is where semantic search provides value over keywords.
        """
        report = await evaluator.evaluate_dataset("cross_domain_transfer_benchmark")

        MINIMUM_CROSS_DOMAIN_RECALL = 0.60

        assert report.recall_at_5 >= MINIMUM_CROSS_DOMAIN_RECALL, (
            f"Cross-domain recall ({report.recall_at_5:.3f}) below minimum. "
            f"Semantic transfer across domains is insufficient."
        )

    # ============================================================
    # Multi-Agent Handoff Benchmark (MAHB) Tests - SHARED Tier
    # ============================================================

    @pytest.mark.edd
    @pytest.mark.shared_tier
    async def test_mahb_cross_agent_retrieval(self, evaluator):
        """
        REQUIREMENT: Agent B must retrieve Agent A's memories with >= 0.85 success

        This validates the SHARED tier functionality.
        """
        report = await evaluator.evaluate_dataset("multi_agent_handoff_benchmark")

        MINIMUM_HANDOFF_SUCCESS = 0.85

        assert report.recall_at_5 >= MINIMUM_HANDOFF_SUCCESS, (
            f"Cross-agent retrieval ({report.recall_at_5:.3f}) below minimum. "
            f"SHARED tier is not functioning correctly."
        )

    @pytest.mark.edd
    @pytest.mark.shared_tier
    async def test_mahb_consistency_across_agents(self, evaluator):
        """
        REQUIREMENT: Same query from different agents returns same results

        Consistency rate must be >= 0.95
        """
        # Run same queries from "different agents" (simulated)
        report = await evaluator.evaluate_dataset(
            "multi_agent_handoff_benchmark",
            num_runs=5  # Simulate 5 different agents
        )

        MINIMUM_CONSISTENCY = 0.95

        assert report.consistency_rate >= MINIMUM_CONSISTENCY, (
            f"Cross-agent consistency ({report.consistency_rate:.3f}) below minimum. "
            f"Different agents getting different results for same query."
        )

    # ============================================================
    # Negative Cases Benchmark (NCB) Tests
    # ============================================================

    @pytest.mark.edd
    async def test_ncb_false_positive_rate(self, evaluator):
        """
        REQUIREMENT: False positive rate must be <= 0.10

        Irrelevant memories should NOT appear in top results.
        """
        report = await evaluator.evaluate_dataset("negative_cases_benchmark")

        # For NCB, precision measures how well we AVOID irrelevant results
        MAXIMUM_FALSE_POSITIVE = 0.10
        false_positive_rate = 1.0 - report.precision_at_5

        assert false_positive_rate <= MAXIMUM_FALSE_POSITIVE, (
            f"False positive rate ({false_positive_rate:.3f}) exceeds maximum. "
            f"Too many irrelevant memories being retrieved."
        )

    # ============================================================
    # Performance Tests
    # ============================================================

    @pytest.mark.edd
    @pytest.mark.performance
    async def test_latency_p99_under_threshold(self, evaluator):
        """
        REQUIREMENT: P99 latency must be < 100ms

        Memory retrieval cannot bottleneck agent execution.
        """
        report = await evaluator.evaluate_dataset("task_similarity_benchmark")

        MAX_P99_LATENCY_MS = 100

        assert report.latency_p99_ms < MAX_P99_LATENCY_MS, (
            f"P99 latency ({report.latency_p99_ms:.1f}ms) exceeds threshold. "
            f"Memory retrieval is too slow."
        )

    @pytest.mark.edd
    @pytest.mark.performance
    @pytest.mark.parametrize("scale", [1000, 10000, 50000])
    async def test_latency_scales_sublinearly(self, evaluator, scale):
        """
        REQUIREMENT: Latency must scale O(log n), not O(n)

        At 50k memories, latency should be < 2x latency at 1k memories.
        """
        # Load memories to target scale
        await evaluator.sut.load_scale_test_data(scale)

        report = await evaluator.evaluate_dataset("task_similarity_benchmark")

        # Store for comparison
        if scale == 1000:
            pytest.baseline_latency = report.latency_p50_ms
        elif scale == 50000:
            MAX_LATENCY_MULTIPLIER = 2.0
            assert report.latency_p50_ms < pytest.baseline_latency * MAX_LATENCY_MULTIPLIER, (
                f"Latency at {scale} memories ({report.latency_p50_ms:.1f}ms) "
                f"exceeds {MAX_LATENCY_MULTIPLIER}x baseline ({pytest.baseline_latency:.1f}ms). "
                f"Search is not scaling efficiently."
            )

    # ============================================================
    # Stochastic Stability Tests
    # ============================================================

    @pytest.mark.edd
    async def test_result_stability_across_runs(self, evaluator):
        """
        REQUIREMENT: Variance coefficient must be < 0.05

        Results should be stable across repeated runs.
        """
        report = await evaluator.evaluate_dataset(
            "task_similarity_benchmark",
            num_runs=10
        )

        MAX_VARIANCE_COEFFICIENT = 0.05

        assert report.variance_coefficient < MAX_VARIANCE_COEFFICIENT, (
            f"Result variance ({report.variance_coefficient:.3f}) exceeds maximum. "
            f"System is not producing stable results."
        )
```

### 5.3 Integration Tests (SHARED Tier)

```python
class TestSharedTierIntegration:
    """
    Integration tests for SHARED tier via MCP.

    These tests validate that multiple ReAgent instances
    can share semantic memory through the MCP server.
    """

    @pytest.fixture
    async def shared_memory_server(self):
        """Start shared MCP server for tests."""
        import subprocess

        proc = subprocess.Popen(
            ["local-faiss-mcp", "--index-dir", "./test_shared_memory"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        yield proc
        proc.terminate()

    @pytest.fixture
    async def agent_a(self, shared_memory_server):
        """Agent A connected to shared memory."""
        return SemanticMemorySystem(
            use_mcp=True,
            mcp_config={"command": "local-faiss-mcp", "args": ["--index-dir", "./test_shared_memory"]}
        )

    @pytest.fixture
    async def agent_b(self, shared_memory_server):
        """Agent B connected to same shared memory."""
        return SemanticMemorySystem(
            use_mcp=True,
            mcp_config={"command": "local-faiss-mcp", "args": ["--index-dir", "./test_shared_memory"]}
        )

    @pytest.mark.integration
    @pytest.mark.shared_tier
    async def test_agent_a_store_agent_b_retrieve(self, agent_a, agent_b):
        """
        SHARED TIER: Agent B can retrieve what Agent A stored.
        """
        # Agent A stores a memory
        await agent_a.store(
            key="research_001",
            content="Analyzed competitor pricing: AWS S3 at $0.023/GB",
            metadata={"agent": "research", "domain": "competitive_analysis"}
        )

        # Agent B searches for related content
        results = await agent_b.search(
            query="What do we know about cloud storage pricing?",
            top_k=5
        )

        # Verify Agent B found Agent A's memory
        assert len(results) >= 1
        assert any("competitor pricing" in r["content"].lower() for r in results)

    @pytest.mark.integration
    @pytest.mark.shared_tier
    async def test_concurrent_access(self, agent_a, agent_b):
        """
        SHARED TIER: Concurrent reads and writes don't corrupt data.
        """
        import asyncio

        # Concurrent writes from both agents
        await asyncio.gather(
            agent_a.store("a_memory_1", "Content from Agent A"),
            agent_b.store("b_memory_1", "Content from Agent B"),
            agent_a.store("a_memory_2", "More content from Agent A"),
            agent_b.store("b_memory_2", "More content from Agent B"),
        )

        # Both agents can retrieve all memories
        a_results = await agent_a.search("content from agent", top_k=10)
        b_results = await agent_b.search("content from agent", top_k=10)

        assert len(a_results) >= 4
        assert len(b_results) >= 4

    @pytest.mark.integration
    @pytest.mark.shared_tier
    async def test_memory_persistence_across_agent_restart(self, shared_memory_server):
        """
        SHARED TIER: Memories persist when individual agents restart.
        """
        # Agent A stores memory and "dies"
        agent_a = SemanticMemorySystem(use_mcp=True, ...)
        await agent_a.store("persistent_memory", "This should survive agent restart")
        del agent_a  # Agent A terminated

        # New Agent C connects to same shared memory
        agent_c = SemanticMemorySystem(use_mcp=True, ...)

        results = await agent_c.search("survive agent restart", top_k=5)

        assert len(results) >= 1
        assert "persistent_memory" in results[0]["key"]
```

---

## 6. Solution Fitness Assessment

### 6.1 Assessment Framework

Before implementing, we assess whether "ReAgent + Semantic Shared Memory via MCP" is the right solution:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Solution Fitness Scorecard                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Problem                          │ Solution Addresses? │ Confidence │ Score │
│  ─────────────────────────────────┼─────────────────────┼────────────┼───────│
│  P1: Memory Retrieval Accuracy    │ Yes (vector search) │ High       │  9/10 │
│  P2: Cross-Domain Transfer        │ Yes (embeddings)    │ Medium     │  7/10 │
│  P3: Multi-Agent Sharing          │ Yes (MCP server)    │ High       │  8/10 │
│  P4: Scalable Search              │ Yes (FAISS index)   │ High       │  9/10 │
│  P5: Context Preservation         │ Partial (metadata)  │ Medium     │  6/10 │
│  ─────────────────────────────────┼─────────────────────┼────────────┼───────│
│                                   │                     │ OVERALL    │ 7.8/10│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Pre-Implementation Validation Checklist

Before implementing, run these lightweight validation experiments:

#### 6.2.1 Embedding Quality Validation
```python
async def validate_embedding_quality():
    """
    Quick validation that chosen embedding model captures task similarity.

    Run BEFORE implementing full system.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Test cases from golden dataset
    test_pairs = [
        # (query, candidate, expected_similarity_bucket)
        ("Analyze Q3 sales", "Review Q3 revenue", "high"),
        ("Debug memory leak", "Fix resource exhaustion", "medium"),
        ("Write unit tests", "Plan team offsite", "low"),
    ]

    results = []
    for query, candidate, expected in test_pairs:
        q_emb = model.encode(query)
        c_emb = model.encode(candidate)
        similarity = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))

        results.append({
            "query": query,
            "candidate": candidate,
            "similarity": similarity,
            "expected": expected,
            "bucket": "high" if similarity > 0.7 else "medium" if similarity > 0.4 else "low"
        })

    # Validate buckets match expectations
    matches = sum(1 for r in results if r["bucket"] == r["expected"])
    accuracy = matches / len(results)

    print(f"Embedding quality validation: {accuracy:.0%} bucket accuracy")
    return accuracy >= 0.80  # 80% threshold
```

#### 6.2.2 MCP Feasibility Validation
```python
async def validate_mcp_feasibility():
    """
    Validate that MCP protocol can support our use case.

    Run BEFORE implementing full system.
    """
    import subprocess
    import json
    import time

    # Start local-faiss-mcp
    proc = subprocess.Popen(
        ["local-faiss-mcp", "--index-dir", "./validation_test"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    # Test round-trip latency
    latencies = []
    for i in range(100):
        start = time.perf_counter()

        # Send MCP request
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "query_rag_store", "arguments": {"query": "test", "top_k": 5}},
            "id": i
        }
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

        # Read response
        response = proc.stdout.readline()

        latencies.append((time.perf_counter() - start) * 1000)

    proc.terminate()

    p99_latency = sorted(latencies)[95]
    print(f"MCP round-trip P99 latency: {p99_latency:.1f}ms")

    return p99_latency < 50  # 50ms threshold
```

### 6.3 Go/No-Go Criteria

| Criterion | Threshold | Validation Method |
|-----------|-----------|-------------------|
| Embedding quality | >= 80% bucket accuracy | `validate_embedding_quality()` |
| MCP latency | < 50ms P99 | `validate_mcp_feasibility()` |
| TSB precision improvement | >= 20% over baseline | Run baseline + semantic on TSB sample |
| CDTB cross-domain recall | >= 50% | Run on CDTB sample |
| MAHB shared retrieval | >= 80% | Run on MAHB sample |

**Decision Rule:**
- 5/5 criteria pass → **GO** - Proceed with implementation
- 4/5 criteria pass → **CONDITIONAL GO** - Proceed with risk mitigation
- 3/5 or fewer pass → **NO GO** - Reassess solution approach

---

## 7. Baseline Comparisons

### 7.1 Baseline: Current ReAgent Keyword Matching

```python
class KeywordBaselineSystem:
    """
    Baseline system using current ReAgent keyword matching.

    Used for A/B comparison with semantic approach.
    """

    def get_similar_executions(
        self,
        task_description: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Current ReAgent implementation (memory.py:204-236).

        Uses Jaccard index on extracted keywords.
        """
        task_keywords = self._extract_keywords(task_description)

        similarities = []
        for key, entry in self.memory_store.items():
            entry_keywords = set(entry.get("keywords", []))

            if task_keywords and entry_keywords:
                common = task_keywords & entry_keywords
                similarity = len(common) / len(task_keywords | entry_keywords)
                similarities.append((key, entry, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[2], reverse=True)

        return [
            {"key": k, "content": e["content"], "similarity": s}
            for k, e, s in similarities[:top_k]
        ]
```

### 7.2 Expected Improvements

| Metric | Baseline (Keyword) | Target (Semantic) | Improvement |
|--------|-------------------|-------------------|-------------|
| Precision@5 | ~0.55 | >= 0.70 | +27% |
| Recall@10 | ~0.60 | >= 0.85 | +42% |
| MRR | ~0.50 | >= 0.75 | +50% |
| Cross-Domain Recall | ~0.20 | >= 0.60 | +200% |
| P99 Latency (10k) | ~500ms | < 100ms | -80% |

### 7.3 A/B Evaluation Protocol

```python
async def run_ab_comparison(dataset_name: str):
    """
    Run A/B comparison between baseline and semantic system.
    """
    evaluator_baseline = EDDEvaluator(KeywordBaselineSystem(), "./golden_datasets")
    evaluator_semantic = EDDEvaluator(SemanticMemorySystem(), "./golden_datasets")

    report_a = await evaluator_baseline.evaluate_dataset(dataset_name)
    report_b = await evaluator_semantic.evaluate_dataset(dataset_name)

    # Statistical comparison
    comparison = {
        "precision_improvement": (report_b.precision_at_5 - report_a.precision_at_5) / report_a.precision_at_5,
        "recall_improvement": (report_b.recall_at_10 - report_a.recall_at_10) / report_a.recall_at_10,
        "mrr_improvement": (report_b.mrr - report_a.mrr) / report_a.mrr,
        "latency_reduction": (report_a.latency_p99_ms - report_b.latency_p99_ms) / report_a.latency_p99_ms,
    }

    # Compute statistical significance
    # ... (paired t-test implementation)

    return comparison
```

---

## 8. Implementation Verification Tests

Once EDD validation passes and implementation proceeds, these deterministic tests verify correctness:

### 8.1 Unit Tests (Traditional TDD)

```python
class TestSemanticMemoryAdapter:
    """Traditional unit tests for implementation correctness."""

    def test_store_creates_embedding(self):
        """Verify content is embedded when stored."""
        adapter = SemanticMemoryAdapter(index_dir="./test")
        adapter.store("key1", "Test content")

        # Verify embedding was created
        assert adapter.store.index.ntotal == 1

    def test_search_returns_ranked_results(self):
        """Verify search returns results in similarity order."""
        adapter = SemanticMemoryAdapter(index_dir="./test")
        adapter.store("key1", "Python programming tutorial")
        adapter.store("key2", "Java enterprise development")
        adapter.store("key3", "Python data science guide")

        results = adapter.search("Python coding", top_k=3)

        # Python results should rank higher than Java
        assert results[0]["key"] in ["key1", "key3"]
        assert results[1]["key"] in ["key1", "key3"]

    def test_mcp_mode_connects_to_server(self):
        """Verify MCP mode establishes connection."""
        adapter = SemanticMemoryAdapter(
            use_mcp=True,
            mcp_config={"command": "local-faiss-mcp", "args": []}
        )

        assert adapter.mcp_client is not None
        assert adapter.mcp_client._connected
```

### 8.2 Contract Tests

```python
class TestMCPContract:
    """Verify MCP protocol contract with local-faiss-mcp."""

    def test_ingest_document_request_format(self):
        """Verify ingest request matches expected MCP format."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "ingest_document",
                "arguments": {
                    "content": "Test content",
                    "source": "test_source"
                }
            },
            "id": 1
        }

        # Validate against MCP schema
        assert validate_mcp_request(request)

    def test_query_rag_store_response_format(self):
        """Verify query response matches expected format."""
        response = {
            "jsonrpc": "2.0",
            "result": {
                "results": [
                    {"text": "...", "distance": 0.5, "source": "..."}
                ]
            },
            "id": 1
        }

        # Validate response structure
        assert "results" in response["result"]
        assert all("distance" in r for r in response["result"]["results"])
```

---

## 9. Test Execution Plan

### 9.1 Execution Order

```
Phase 0: Pre-Implementation Validation (Before any code)
├── Run validate_embedding_quality()
├── Run validate_mcp_feasibility()
├── Sample evaluation on TSB (100 queries)
├── Sample evaluation on CDTB (50 queries)
└── GO/NO-GO decision

Phase 1: Golden Dataset Creation (Parallel with Phase 0)
├── Create TSB dataset (200 queries)
├── Create CDTB dataset (100 queries)
├── Create MAHB dataset (150 scenarios)
├── Create NCB dataset (100 queries)
└── Human labeling and validation

Phase 2: Implementation + Continuous Evaluation
├── Implement SemanticMemoryAdapter
│   └── Run unit tests (TDD)
├── Implement MCP client
│   └── Run contract tests
├── Run EDD evaluation suite
│   └── Must pass all thresholds
└── Iterate until EDD tests pass

Phase 3: Integration Testing
├── SHARED tier integration tests
├── Multi-agent scenarios
├── Performance at scale
└── A/B comparison with baseline

Phase 4: Production Validation
├── Shadow mode deployment
├── Monitor real-world metrics
└── Gradual rollout
```

### 9.2 CI/CD Integration

```yaml
# .github/workflows/edd-evaluation.yml
name: EDD Evaluation Suite

on:
  push:
    paths:
      - 'reagent/core/semantic_memory.py'
      - 'reagent/core/memory.py'

jobs:
  edd-evaluation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e .[semantic,test]

      - name: Run EDD Evaluation Suite
        run: pytest tests/edd/ -m "edd" --tb=short
        env:
          GOLDEN_DATASETS_DIR: ./golden_datasets

      - name: Upload Evaluation Report
        uses: actions/upload-artifact@v3
        with:
          name: edd-report
          path: ./edd_reports/
```

---

## 10. Summary

### 10.1 EDD Approach Benefits

1. **Quality-Focused:** Tests validate the *quality* of outputs, not just correctness
2. **Stochastic-Aware:** Metrics account for LLM non-determinism
3. **Problem-First:** Started with problem space, validated solution fit
4. **Baseline-Comparative:** Always comparing against current system
5. **Threshold-Driven:** Clear pass/fail criteria for go/no-go decisions

### 10.2 Key Thresholds Summary

| Test | Metric | Threshold |
|------|--------|-----------|
| TSB Precision | Precision@5 | >= 0.66 (20% over baseline) |
| TSB Recall | Recall@10 | >= 0.80 |
| TSB Ranking | MRR | >= 0.70 |
| CDTB Transfer | Cross-domain Recall@5 | >= 0.60 |
| MAHB Sharing | Cross-agent Recall@5 | >= 0.85 |
| MAHB Consistency | Consistency Rate | >= 0.95 |
| NCB Precision | False Positive Rate | <= 0.10 |
| Performance | P99 Latency | < 100ms |
| Stability | Variance Coefficient | < 0.05 |

### 10.3 Next Steps

1. **Create Golden Datasets** - Priority: TSB and MAHB first
2. **Run Pre-Implementation Validation** - Embedding quality + MCP feasibility
3. **Make GO/NO-GO Decision** - Based on validation results
4. **Implement with Continuous EDD Evaluation** - Every PR runs EDD suite

---

*Document generated: 2025-01-14*
*Last updated: 2025-01-14*
*Author: AI Research Agent*
*Version: 1.1*
*Methodology: Evaluation Driven Development (EDD)*

**Changelog:**
- v1.1: Added open-source dataset sourcing research with specific datasets, URLs, and sourcing scripts
- v1.0: Initial EDD test plan with dataset structures and evaluation methodology
