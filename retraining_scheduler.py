"""Scheduled retraining job using APScheduler.

Runs a retraining cycle on a configurable schedule and writes a status file
so the Streamlit monitoring page can surface the last run outcome.

Configuration via environment variables
----------------------------------------
RETRAIN_MODEL_TYPE      Model to retrain: random_forest | xgboost | neural_network
                        Default: random_forest
RETRAIN_TASK_TYPE       Task type: classification | regression
                        Default: classification
RETRAIN_SCHEDULE        Schedule type: interval | cron
                        Default: interval
RETRAIN_INTERVAL_HOURS  Hours between runs (interval schedule)
                        Default: 24
RETRAIN_CRON            Cron expression for cron schedule, e.g. "0 2 * * *"
                        Default: "0 2 * * *"  (2 AM every day)
RETRAIN_CV_FOLDS        Cross-validation folds
                        Default: 5
RETRAIN_AUTO_PROMOTE    Whether to auto-promote if improved: true | false
                        Default: false
RETRAIN_REGISTRY_NAME   MLflow registry model name (required for auto-promote)
                        Default: automl-model
MLFLOW_TRACKING_URI     MLflow server URI
                        Default: http://localhost:5001
MLFLOW_EXPERIMENT       MLflow experiment name
                        Default: automl-experiments
DATA_DIR                Directory containing processed CSV splits
                        Default: data/processed
ARTIFACT_DIR            Directory for model joblib files
                        Default: artifacts
STATUS_FILE             Path to write scheduler status JSON
                        Default: artifacts/scheduler_status.json

Usage
-----
    python retraining_scheduler.py

Run in the background (Unix):
    nohup python retraining_scheduler.py > logs/scheduler.log 2>&1 &
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("retraining_scheduler")


# ── Configuration from environment ────────────────────────────────────────────

MODEL_TYPE: str = os.environ.get("RETRAIN_MODEL_TYPE", "random_forest")
TASK_TYPE: str = os.environ.get("RETRAIN_TASK_TYPE", "classification")
SCHEDULE: str = os.environ.get("RETRAIN_SCHEDULE", "interval")
INTERVAL_HOURS: float = float(os.environ.get("RETRAIN_INTERVAL_HOURS", "24"))
CRON_EXPR: str = os.environ.get("RETRAIN_CRON", "0 2 * * *")
CV_FOLDS: int = int(os.environ.get("RETRAIN_CV_FOLDS", "5"))
AUTO_PROMOTE: bool = os.environ.get("RETRAIN_AUTO_PROMOTE", "false").lower() == "true"
REGISTRY_NAME: str = os.environ.get("RETRAIN_REGISTRY_NAME", "automl-model")
MLFLOW_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT: str = os.environ.get("MLFLOW_EXPERIMENT", "automl-experiments")
DATA_DIR: str = os.environ.get("DATA_DIR", "data/processed")
ARTIFACT_DIR: str = os.environ.get("ARTIFACT_DIR", "artifacts")
STATUS_FILE: str = os.environ.get("STATUS_FILE", "artifacts/scheduler_status.json")


# ── Status helpers ─────────────────────────────────────────────────────────────

def _write_status(
    status: str,
    last_run: str | None = None,
    last_result: dict | None = None,
    error: str | None = None,
) -> None:
    """Write scheduler status to JSON file for the UI to read."""
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    payload = {
        "status": status,
        "model_type": MODEL_TYPE,
        "task_type": TASK_TYPE,
        "schedule": SCHEDULE,
        "interval_hours": INTERVAL_HOURS if SCHEDULE == "interval" else None,
        "cron": CRON_EXPR if SCHEDULE == "cron" else None,
        "auto_promote": AUTO_PROMOTE,
        "registry_name": REGISTRY_NAME if AUTO_PROMOTE else None,
        "last_run": last_run,
        "last_result": last_result,
        "error": error,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(STATUS_FILE, "w") as fh:
        json.dump(payload, fh, indent=2)


# ── Retraining job ─────────────────────────────────────────────────────────────

def _retrain_job() -> None:
    """Single retraining execution — called by APScheduler."""
    started_at = datetime.now().isoformat(timespec="seconds")
    logger.info(
        "Retraining job started — model=%s, task=%s", MODEL_TYPE, TASK_TYPE
    )
    _write_status("running", last_run=started_at)

    try:
        from src.pipelines.retraining import run_retraining

        result = run_retraining(
            model_type=MODEL_TYPE,
            task_type=TASK_TYPE,
            cv_folds=CV_FOLDS,
            experiment_name=MLFLOW_EXPERIMENT,
            tracking_uri=MLFLOW_URI,
            registry_name=REGISTRY_NAME if AUTO_PROMOTE else None,
            auto_promote=AUTO_PROMOTE,
            data_dir=DATA_DIR,
            artifact_dir=ARTIFACT_DIR,
            log_callback=logger.info,
        )

        pk = result.primary_metric
        new_score = result.new_test_metrics.get(pk, 0.0)
        old_score = (
            result.old_test_metrics.get(pk, 0.0)
            if result.old_test_metrics else None
        )
        logger.info(
            "Retraining complete — %s: new=%.4f, old=%s, improved=%s, promoted=%s",
            pk,
            new_score,
            f"{old_score:.4f}" if old_score is not None else "N/A",
            result.improved,
            result.promoted,
        )

        _write_status(
            "idle",
            last_run=started_at,
            last_result=result.as_dict(),
        )

    except Exception as exc:
        logger.exception("Retraining job failed: %s", exc)
        _write_status("error", last_run=started_at, error=str(exc))


# ── Scheduler setup ────────────────────────────────────────────────────────────

def _build_scheduler():
    """Build and return a configured APScheduler BlockingScheduler."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone="UTC")

    if SCHEDULE == "cron":
        # Parse "minute hour dom month dow" into keyword args
        parts = CRON_EXPR.split()
        if len(parts) != 5:
            raise ValueError(
                f"RETRAIN_CRON must be a 5-field cron expression, got: {CRON_EXPR!r}"
            )
        minute, hour, day, month, day_of_week = parts
        scheduler.add_job(
            _retrain_job,
            "cron",
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            id="retrain",
            replace_existing=True,
        )
        logger.info("Cron schedule: %s (UTC)", CRON_EXPR)
    else:
        scheduler.add_job(
            _retrain_job,
            "interval",
            hours=INTERVAL_HOURS,
            id="retrain",
            replace_existing=True,
        )
        logger.info("Interval schedule: every %.1f hour(s)", INTERVAL_HOURS)

    return scheduler


def _handle_shutdown(signum, frame) -> None:  # noqa: ANN001
    logger.info("Received signal %s — shutting down scheduler.", signum)
    _write_status("stopped")
    sys.exit(0)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    logger.info(
        "Starting retraining scheduler — model=%s, task=%s, schedule=%s",
        MODEL_TYPE, TASK_TYPE, SCHEDULE,
    )

    _write_status("starting")

    scheduler = _build_scheduler()

    # Run one cycle immediately on startup so we have a baseline result
    run_now = os.environ.get("RETRAIN_RUN_ON_START", "false").lower() == "true"
    if run_now:
        logger.info("RETRAIN_RUN_ON_START=true — running retraining immediately.")
        _retrain_job()

    _write_status("idle")
    logger.info("Scheduler ready. Waiting for next trigger…")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        _write_status("stopped")
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
