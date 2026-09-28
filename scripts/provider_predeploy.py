import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_manager import ModelManager  # noqa: E402


async def main() -> None:
    manager = ModelManager(Path("config/models.yaml"))
    await manager.validate_provider()
    configured = {manager.model_id(tier) for tier in manager.available_tiers}
    print(
        f"Provider validation passed: {manager.provider}; "
        f"{len(configured)} configured model ID(s) available."
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"Provider validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
