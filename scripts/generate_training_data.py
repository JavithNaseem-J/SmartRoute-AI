import asyncio
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from src.models.nvidia_model import NvidiaModel
from src.utils.logger import logger

# Number of queries to generate per complexity class
SAMPLES_PER_CLASS = 200

# Classes mapping for our classifier (0=simple, 1=medium, 2=complex)
COMPLEXITY_CLASSES = {"simple": 0, "medium": 1, "complex": 2}

SYSTEM_PROMPT = """Generate exactly {num_samples} realistic queries for "{complexity}".
Output exactly a JSON list of strings: ["Q1", "Q2"].
Simple: Basic facts, definitions, small summaries.
Medium: Short code, comparisons, guides.
Complex: Deep architecture, multi-step math, system design."""


async def generate_class(model: NvidiaModel, complexity: str, num_samples: int) -> list[str]:
    logger.info(f"Generating {num_samples} {complexity} queries...")

    prompt = SYSTEM_PROMPT.format(num_samples=num_samples, complexity=complexity)

    try:
        response = await model.agenerate(prompt)

        # Clean up output to extract JSON
        content = response["text"].strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        queries = json.loads(content)

        if not isinstance(queries, list):
            raise ValueError("Expected a list of strings")

        return queries[:num_samples]
    except Exception as e:
        logger.error(f"Failed to generate {complexity} queries: {e}")
        return []


async def main():
    logger.info("Starting synthetic data generation...")

    # Initialize the fast model to use for generation
    model = NvidiaModel("llama-3.1-8b-instant")

    out_dir = Path("data/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "synthetic_queries.csv"

    all_queries = []

    for complexity, class_id in COMPLEXITY_CLASSES.items():
        # Generate in chunks to avoid context limits
        chunk_size = 50
        chunks = SAMPLES_PER_CLASS // chunk_size
        remainder = SAMPLES_PER_CLASS % chunk_size

        sizes = [chunk_size] * chunks
        if remainder > 0:
            sizes.append(remainder)

        for size in sizes:
            queries = await generate_class(model, complexity, size)
            for q in queries:
                all_queries.append({"query": q.replace("\n", " ").strip(), "complexity": class_id})

    # Save to CSV
    logger.info(f"Saving {len(all_queries)} queries to {out_file}")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "complexity"])
        writer.writeheader()
        writer.writerows(all_queries)

    logger.info("Data generation complete!")


if __name__ == "__main__":
    asyncio.run(main())
