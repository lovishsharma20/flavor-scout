"""
Reddit ingestion for Flavor Scout (Step 1).

Fetches a small test set of posts/comments via PRAW and writes
data/raw/raw_comments.csv for downstream cleaning.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
from dotenv import load_dotenv
from praw.exceptions import PRAWException
from prawcore.exceptions import PrawcoreException

from config import (
    COMMENTS_PER_POST,
    POSTS_PER_QUERY,
    QUERIES,
    SORT,
    SUBREDDITS,
    TARGET_TOTAL,
    TIME_FILTER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "raw_comments.csv"
COLUMNS = [
    "source",
    "post_id",
    "comment_id",
    "text",
    "subreddit",
    "timestamp",
    "url",
    "engagement",
]


def load_reddit_client() -> praw.Reddit:
    """Build a PRAW client from environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    missing = [
        name
        for name, value in [
            ("REDDIT_CLIENT_ID", client_id),
            ("REDDIT_CLIENT_SECRET", client_secret),
            ("REDDIT_USER_AGENT", user_agent),
        ]
        if not value or value.startswith("your_")
    ]
    if missing:
        raise ValueError(
            "Missing Reddit credentials in .env: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in your API values."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def _to_iso(utc_timestamp: float) -> str:
    return datetime.fromtimestamp(utc_timestamp, tz=timezone.utc).isoformat()


def _post_row(submission) -> dict:
    text = (submission.title or "").strip()
    body = (submission.selftext or "").strip()
    if body:
        text = f"{text}\n\n{body}".strip()

    return {
        "source": "reddit_post",
        "post_id": submission.id,
        "comment_id": "",
        "text": text,
        "subreddit": str(submission.subreddit),
        "timestamp": _to_iso(submission.created_utc),
        "url": f"https://www.reddit.com{submission.permalink}",
        "engagement": int(submission.score or 0),
    }


def _comment_row(submission, comment) -> dict:
    return {
        "source": "reddit_comment",
        "post_id": submission.id,
        "comment_id": comment.id,
        "text": (comment.body or "").strip(),
        "subreddit": str(submission.subreddit),
        "timestamp": _to_iso(comment.created_utc),
        "url": f"https://www.reddit.com{comment.permalink}",
        "engagement": int(comment.score or 0),
    }


def collect_rows(reddit: praw.Reddit) -> list[dict]:
    """Search configured queries and collect posts + top-level comments."""
    rows: list[dict] = []
    seen_ids: set[str] = set()
    subreddit_filter = "+".join(SUBREDDITS)

    for query in QUERIES:
        if len(rows) >= TARGET_TOTAL:
            break

        logger.info("Searching query=%r in subreddits=%s", query, subreddit_filter)
        try:
            results = reddit.subreddit(subreddit_filter).search(
                query,
                sort=SORT,
                time_filter=TIME_FILTER,
                limit=POSTS_PER_QUERY,
            )
        except (PRAWException, PrawcoreException) as exc:
            logger.error("Search failed for query=%r: %s", query, exc)
            continue

        try:
            for submission in results:
                if len(rows) >= TARGET_TOTAL:
                    break

                post_key = f"post:{submission.id}"
                if post_key not in seen_ids:
                    row = _post_row(submission)
                    if row["text"]:
                        rows.append(row)
                        seen_ids.add(post_key)

                try:
                    submission.comments.replace_more(limit=0)
                    comments = submission.comments.list()[:COMMENTS_PER_POST]
                except (PRAWException, PrawcoreException) as exc:
                    logger.warning(
                        "Could not load comments for post=%s: %s",
                        submission.id,
                        exc,
                    )
                    continue

                for comment in comments:
                    if len(rows) >= TARGET_TOTAL:
                        break
                    # Skip nested MoreComments leftovers and deleted/empty bodies
                    if not hasattr(comment, "body"):
                        continue
                    body = (comment.body or "").strip()
                    if not body or body in {"[deleted]", "[removed]"}:
                        continue

                    comment_key = f"comment:{comment.id}"
                    if comment_key in seen_ids:
                        continue

                    rows.append(_comment_row(submission, comment))
                    seen_ids.add(comment_key)
        except (PRAWException, PrawcoreException) as exc:
            logger.error("Error while iterating results for query=%r: %s", query, exc)
            continue

    return rows


def save_raw_csv(rows: list[dict], output_path: Path = RAW_OUTPUT_PATH) -> Path:
    """Persist collected rows to CSV with a stable column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Wrote %s rows to %s", len(df), output_path)
    return output_path


def run() -> Path:
    reddit = load_reddit_client()
    rows = collect_rows(reddit)
    if not rows:
        raise RuntimeError(
            "No Reddit rows collected. Check credentials, network, "
            "and config queries/subreddits."
        )
    if len(rows) < 50:
        logger.warning(
            "Collected only %s rows (target was %s). Consider widening queries.",
            len(rows),
            TARGET_TOTAL,
        )
    return save_raw_csv(rows)


def main() -> None:
    try:
        output = run()
        print(f"Success: saved raw Reddit data to {output}")
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except (PRAWException, PrawcoreException) as exc:
        logger.error("Reddit API error: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("Unexpected failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
