"""kappa_schemas.py — JSON Schema definitions for the Kappa pipeline.

Producers (experiment_pusher, session_archiver) idempotently `PUT` their
schema document to `/_schemas/<name>/v<N>` in Firebase RTDB on init. A
record carrying `schema_version: N` (or a manifest carrying `version: N`,
or an envelope carrying `_v: N`) is *self-locating*: the consumer reads
the version, fetches the schema doc, and parses with confidence.

Conventions:
- Each schema is a JSON Schema draft-2020-12 object.
- Bump the **version integer** in `SCHEMAS[<name>]["v"]` on a breaking
  change. Add fields → forward-compat, no bump (just amend the dict).
  Remove or rename fields → breaking, bump version, write a new entry.
- Producers always PUT the *latest* known version's schema doc on init.
  Old version docs in RTDB stay readable by old consumers.

Why JSON Schema (not protobuf):
- Curl-able / jq-able from anywhere; no toolchain.
- Lives in Firebase next to the data, no separate registry.
- The data plane stays JSON; we only describe it, we don't re-encode it.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# v1 schemas — the format shipped 2026-04-30
# ---------------------------------------------------------------------------

KAPPA_MANIFEST_V1 = {
    "dialect": "json-schema-2020-12",
    "title": "Kappa pipeline manifest",
    "description": (
        "Written to <run_dir>/_kappa_manifest.json on pusher init. "
        "Lets kappa_packer / readers find experiment_id, run_id, and "
        "Firebase URL from any local run directory."
    ),
    "type": "object",
    "required": [
        "version", "experiment_id", "run_id", "kind",
        "firebase_url", "hf_user", "hf_bucket",
    ],
    "properties": {
        "version":       {"type": "integer", "const": 1},
        "experiment_id": {"type": "string"},
        "run_id":        {"type": "string"},
        "kind":          {"type": "string"},
        "firebase_url":  {"type": "string", "format": "uri"},
        "hf_user":       {"type": "string"},
        "hf_bucket":     {"type": "string"},
    },
    "additionalProperties": False,
}


STREAM_META_V1 = {
    "dialect": "json-schema-2020-12",
    "title": "Kappa stream meta document",
    "description": (
        "Stored at /streams_meta/<exp>/<run>/<stream>. URL templates "
        "live here once per stream; records carry no URL strings. "
        "Counters drive the dashboard's pack-progress display."
    ),
    "type": "object",
    "required": [
        "schema_version", "stream", "experiment_id", "run_id", "kind",
        "hf_user", "hf_bucket", "prefix",
        "url_browse_template", "url_hfsync_template",
        "pack_threshold_bytes", "pack_threshold_records",
        "pack_threshold_hours", "current_size_bytes",
        "current_record_count", "pack_progress_pct", "shard_started_at",
    ],
    "properties": {
        "schema_version":          {"type": "integer", "const": 1},
        "stream":                  {"type": "string"},
        "experiment_id":           {"type": "string"},
        "run_id":                  {"type": "string"},
        "kind":                    {"type": "string"},
        "hf_user":                 {"type": "string"},
        "hf_bucket":               {"type": "string"},
        "prefix":                  {"type": "string"},
        "url_browse_template":     {"type": "string"},
        "url_hfsync_template":     {"type": "string"},
        "pack_threshold_bytes":    {"type": "integer", "minimum": 1},
        "pack_threshold_records":  {"type": "integer", "minimum": 1},
        "pack_threshold_hours":    {"type": "number",  "minimum": 0},
        "current_size_bytes":      {"type": "integer", "minimum": 0},
        "current_record_count":    {"type": "integer", "minimum": 0},
        "pack_progress_pct":       {"type": "number",  "minimum": 0},
        "last_pack_at":            {"type": ["number", "null"]},
        "last_pack_filename":      {"type": ["string", "null"]},
        "shard_started_at":        {"type": "number"},
    },
    "additionalProperties": False,
}


SESSION_ENVELOPE_V1 = {
    "dialect": "json-schema-2020-12",
    "title": "Claude Code session record envelope",
    "description": (
        "Written by session_archiver.py around each Claude Code session "
        "record. The original record is JSON-encoded into _payload as a "
        "single string column so heterogeneous record shapes don't break "
        "Parquet column inference."
    ),
    "type": "object",
    "required": ["_v", "_id", "ts", "type", "session_id", "_payload"],
    "properties": {
        "_v":         {"type": "integer", "const": 1},
        "_id":        {"type": "string", "description":
                       "stable record id (uuid / leafUuid / messageId / synthetic hash)"},
        "ts":         {"type": "number", "description": "epoch seconds"},
        "type":       {"type": "string"},
        "session_id": {"type": "string"},
        "_payload":   {"type": "string", "description":
                       "JSON-encoded original Claude Code record"},
    },
    "additionalProperties": True,  # ts gets injected by pusher.stream()
}


# Schema registry — name → (version, schema_doc).
# Producers iterate this on init and PUT each entry to /_schemas/<name>/v<version>.
SCHEMAS: dict[str, dict] = {
    "kappa_manifest":   {"v": 1, "doc": KAPPA_MANIFEST_V1},
    "stream_meta":      {"v": 1, "doc": STREAM_META_V1},
    "session_envelope": {"v": 1, "doc": SESSION_ENVELOPE_V1},
}


def schema_path(name: str, version: int) -> str:
    """RTDB path for a versioned schema document."""
    return f"_schemas/{name}/v{version}"
