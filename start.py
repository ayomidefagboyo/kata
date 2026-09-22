#!/usr/bin/env python3
"""
Kata Autonomous Perpetual Futures Agent - Runtime Launcher

Usage:
    python start.py [web|signals-worker|learning-worker|all]

Default is 'web'.
"""
import os
import sys
import argparse
import asyncio
import logging

# Scientific libraries can otherwise reserve one native worker per CPU. On a
# cloud container or VM that leaves too few process/thread slots for FastAPI's
# blocking-I/O executor.
for thread_limit_env in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_limit_env, "1")

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(repo_root, ".env"))

from kata.config.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("kata.launcher")


def run_web():
    """Run the FastAPI web application and static client."""
    import uvicorn

    logger.info("=" * 60)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Mode: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"Health Check: http://{settings.HOST}:{settings.PORT}/health")
    logger.info(f"Web Cockpit:  http://{settings.HOST}:{settings.PORT}/")
    logger.info(f"API Docs:     http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("=" * 60)

    uvicorn.run(
        "kata.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        timeout_keep_alive=30,
        limit_concurrency=100,
    )


def run_signals_worker():
    """Run the platform signals background worker."""
    logger.info("Starting Platform Signals Worker...")
    from kata.workers.platform_signals_worker import main as worker_main
    sys.exit(0 if asyncio.run(worker_main()) else 1)


def run_learning_worker():
    """Run the multi-level learning background worker."""
    logger.info("Starting Multi-Level Learning Worker...")
    from kata.workers.multi_level_learning_worker import get_multi_level_learning_worker
    worker = get_multi_level_learning_worker()

    async def _runner():
        await worker.start()
        while worker.is_running:
            await asyncio.sleep(1)

    try:
        asyncio.run(_runner())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping Multi-Level Learning Worker...")
        asyncio.run(worker.stop())


def run_all():
    """Run web service with background workers in a single process (for local dev / single container)."""
    import uvicorn
    from kata.workers.multi_level_learning_worker import get_multi_level_learning_worker

    logger.info("Starting Kata in combined mode (Web + Learning Loop)...")
    learning_worker = get_multi_level_learning_worker()

    # Hook learning worker into the event loop once uvicorn starts
    import kata.main
    original_lifespan = kata.main.lifespan

    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def combined_lifespan(app):
        async with original_lifespan(app):
            learning_task = asyncio.create_task(learning_worker.start())
            logger.info("✅ Multi-Level Learning Worker launched inside web process")
            yield
            await learning_worker.stop()
            learning_task.cancel()

    kata.main.app.router.lifespan_context = combined_lifespan

    run_web()


def main():
    parser = argparse.ArgumentParser(description="Kata Autonomous Perpetual Agent Launcher")
    parser.add_argument(
        "target",
        nargs="?",
        default="web",
        choices=["web", "signals-worker", "learning-worker", "all"],
        help="Service target to run (default: web)"
    )
    args = parser.parse_args()

    if args.target == "web":
        run_web()
    elif args.target == "signals-worker":
        run_signals_worker()
    elif args.target == "learning-worker":
        run_learning_worker()
    elif args.target == "all":
        run_all()


if __name__ == "__main__":
    main()
