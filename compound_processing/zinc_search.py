from __future__ import annotations

"""Legacy compatibility wrapper for the historical ZINC search module.

New code should import from `compound_processing.compound_database_search`.
This module re-exports the old public entry points so existing callers keep
working without changes.
"""

from compound_processing.compound_database_search import *  # noqa: F401,F403
