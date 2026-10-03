#!/usr/bin/env python3
"""Maintainer-only capture of historical retrieval-set digests and source hashes.

No calculations, external writes, molecular structures, or uploads are involved.
Ordinary users do not need the original notebook workspace.
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

from package.common import ROOT, SEEDS, sha256, write_json
from package.validation import set_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, required=True)
    args = parser.parse_args()
    csv.field_size_limit(sys.maxsize)
    out = ROOT / "reference_data/cohorts/retrieved_set_digests.csv"
    fields = ["seed", "target_id", "method_label", "percentile", "active_digest", "inactive_digest"]
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed in SEEDS:
            source = args.source_workspace / f"resultados_EF_repeticiones_con_inactivos_10_90/seed_{seed}/retrieved_active_sets_all_methods.csv"
            with source.open(newline="", encoding="utf-8") as data:
                for row in csv.DictReader(data):
                    writer.writerow({"seed":seed, "target_id":row["target_id"],
                        "method_label":row["method_label"], "percentile":row["percentile"],
                        "active_digest":set_digest(ast.literal_eval(row["retrieved_active_ids"])),
                        "inactive_digest":set_digest(ast.literal_eval(row["retrieved_inactive_ids"]))})
    references = ROOT / "reference_data"
    write_json(ROOT / "reference_lock.json", {"files":[
        {"path":str(p.relative_to(references)), "size_bytes":p.stat().st_size, "sha256":sha256(p)}
        for p in sorted(references.rglob("*")) if p.is_file()]})
    write_json(ROOT / "engine/SOURCE_CAPTURE.json", {"origin":"evaluation_scripts in the historical evaluation workspace",
        "scientific_dependencies":"engine/dependencies/SOURCES.json",
        "adapter_changes":["Frozen BLAST ranking input with the full benchmark exclusion universe",
            "Adjacent vendored kernel imports instead of legacy live-repository sys.path entries",
            "Standalone sensitivity renderer extracted without changing plotting statements",
            "Historical union category JSON in place of notebook parsing",
            "Optional BSI provenance entry for this non-BSI figure package"],
        "files":{str(p.relative_to(ROOT / 'engine')):sha256(p) for p in sorted((ROOT / 'engine').rglob('*.py'))}})
    print("Captured historical retrieval-set digests and immutable reference/source manifests.")


if __name__ == "__main__":
    main()
