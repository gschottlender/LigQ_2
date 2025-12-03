#!/usr/bin/env python

import os
import json
import argparse
from datetime import date

from huggingface_hub import HfApi, hf_hub_download

from generate_databases.pdb_db import (
    generate_pdb_database,
    update_pdb_database_from_dir,
)
from generate_databases.chembl_db import generate_chembl_database
from generate_databases.uniprot_db import (
    update_target_sequences_pickle,
    uniprot_dict_to_fasta,
)
from compound_processing.compound_helpers import (
    unify_pdb_chembl,
    build_ligand_index,
    build_morgan_representation,
    LigandStore,
)

from generate_databases.merge_pdb_chembl import merge_databases


# HuggingFace dataset repo with the preprocessed databases and initial metadata
HF_DATASET_REPO_ID = "gschottlender/LigQ_2"


# ----------------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full pipeline: PDB + ChEMBL → merged DB → UniProt sequences."
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="databases",
        help="Root directory where processed databases will be stored (PDB, ChEMBL, merged, sequences).",
    )

    parser.add_argument(
        "--temp-data-dir",
        default="temp_data",
        help="Directory for temporary files (e.g. ChEMBL SQLite tarball and extraction).",
    )

    parser.add_argument(
        "--chembl-version",
        type=int,
        default=36,
        help="ChEMBL version to use when regenerating the local database if needed.",
    )

    parser.add_argument(
        "--tanimoto-curation-threshold",
        type=float,
        default=0.35,
        help="Tanimoto threshold for curating 'possible' ChEMBL ligands when merging PDB–ChEMBL.",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Download / sync base databases from HuggingFace
# ----------------------------------------------------------------------
def download_base_databases_from_huggingface(output_dir: str) -> None:
    """
    Sync the preprocessed database files from the HuggingFace dataset repo
    into `output_dir`.

    This repo is assumed to contain:
      - db_metadata.json
      - preprocessed PDB database under output_dir/pdb
      - preprocessed ChEMBL database under output_dir/chembl
      - preprocessed Uniprot sequences database under output_dir/sequences

    huggingface_hub will handle local caching, so repeated runs will not
    re-download all data from scratch.
    """
    api = HfApi()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Syncing preprocessed databases from HF dataset: {HF_DATASET_REPO_ID}")
    files = api.list_repo_files(repo_id=HF_DATASET_REPO_ID, repo_type="dataset")

    for filename in files:
        # This preserves subdirectory structure inside output_dir
        hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )

    print("[INFO] HuggingFace dataset sync completed.")


def main():
    args = parse_args()

    output_dir = args.output_dir
    temp_data_dir = args.temp_data_dir
    chembl_version = args.chembl_version

    # Check whether the output_dir already existed BEFORE this run
    output_dir_already_existed = os.path.isdir(output_dir)

    # Ensure root directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Sync base dataset from HuggingFace ONLY if there is no local DB
    # ------------------------------------------------------------------
    # Logic:
    #   - If output_dir did not exist before, we assume there is no local
    #     snapshot and we need to fetch the initial one from HF.
    #   - If it already existed, we trust the local copy and DO NOT overwrite
    #     it with the (possibly older) HF snapshot.
    metadata_path = os.path.join(output_dir, "db_metadata.json")

    if not output_dir_already_existed and not os.path.exists(metadata_path):
        print(
            "[INFO] No local database snapshot detected. "
            "Downloading base dataset from HuggingFace..."
        )
        download_base_databases_from_huggingface(output_dir)
    else:
        print(
            "[INFO] Local database directory already exists. "
            "Skipping HuggingFace base download."
        )

    # ------------------------------------------------------------------
    # 2) Load metadata (which should now exist in output_dir)
    # ------------------------------------------------------------------
    metadata_path = os.path.join(output_dir, "db_metadata.json")

    if os.path.exists(metadata_path):
        print(f"[INFO] Loading metadata from {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # ------------------------------------------------------------------
    # 3) Update / generate PDB database
    # ------------------------------------------------------------------
    pdb_db_dir = os.path.join(output_dir, "pdb")
    os.makedirs(pdb_db_dir, exist_ok=True)

    # NOTE:
    # - update_pdb_database_from_dir() is assumed to:
    #   * work on top of the preprocessed PDB database retrieved from HF
    #   * incorporate new PDB entries if available in the local raw source
    print("[INFO] Updating PDB database from local PDB directory...")
    update_pdb_database_from_dir(pdb_db_dir, temp_dir=temp_data_dir)
    metadata["pdb_last_update"] = date.today().isoformat()

    # ------------------------------------------------------------------
    # 4) Update / generate ChEMBL database
    # ------------------------------------------------------------------
    chembl_sql_dir = os.path.join(temp_data_dir, "chembl_sql")
    os.makedirs(chembl_sql_dir, exist_ok=True)

    current_chembl_in_metadata = metadata.get("chembl_version")

    # We only regenerate the ChEMBL database if the target version
    # differs from the one stored in metadata.
    need_chembl_update = (
        current_chembl_in_metadata is None
        or current_chembl_in_metadata != chembl_version
    )

    if need_chembl_update:
        print(
            f"[INFO] Regenerating ChEMBL database. "
            f"Metadata version: {current_chembl_in_metadata}, "
            f"target version: {chembl_version}"
        )

        chembl_file = os.path.join(
            chembl_sql_dir, f"chembl_{chembl_version}_sqlite.tar.gz"
        )

        # Remove old tar.gz (e.g. truncated downloads from previous runs)
        if os.path.exists(chembl_file):
            print(f"[INFO] Removing previous tarball: {chembl_file}")
            os.remove(chembl_file)

        # Download ChEMBL SQLite for the requested version
        url = (
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
            f"chembl_{chembl_version}/chembl_{chembl_version}_sqlite.tar.gz"
        )

        print(f"[INFO] Downloading ChEMBL SQLite from: {url}")
        ret = os.system(f"wget -q -P {chembl_sql_dir} {url}")
        if ret != 0:
            raise RuntimeError(
                f"wget failed when downloading ChEMBL SQLite (exit code {ret}). "
                f"Check your network connection and the URL: {url}"
            )

        print("[INFO] Extracting ChEMBL SQLite tarball...")
        ret = os.system(f"tar -xvzf {chembl_file} -C {chembl_sql_dir}")
        if ret != 0:
            raise RuntimeError(
                f"tar extraction failed for {chembl_file} (exit code {ret})."
            )

        # Locate the .db file inside the extracted directory tree
        db_filename = f"chembl_{chembl_version}.db"
        chembl_db_path = None
        for root, _, files in os.walk(chembl_sql_dir):
            if db_filename in files:
                chembl_db_path = os.path.join(root, db_filename)
                break

        if chembl_db_path is None:
            raise FileNotFoundError(
                f"Could not find {db_filename} inside {chembl_sql_dir} "
                "after extraction."
            )

        print(f"[INFO] Generating local ChEMBL database from: {chembl_db_path}")
        chembl_output_dir = os.path.join(output_dir, "chembl")
        os.makedirs(chembl_output_dir, exist_ok=True)

        generate_chembl_database(
            chembl_db_path=chembl_db_path,
            output_dir=chembl_output_dir,
        )

        metadata["chembl_version"] = chembl_version
    else:
        print(
            f"[INFO] ChEMBL database already at version {chembl_version}, "
            "no regeneration needed."
        )

    # ------------------------------------------------------------------
    # 5) Merge PDB + ChEMBL
    # ------------------------------------------------------------------
    # Assumption: merge_databases() expects a directory that contains
    # subfolders "pdb" and "chembl" with the processed data.

    # This step also generates the vector database of compounds from PDB and ChEMBL 
    data_dir = output_dir

    print("[INFO] Merging PDB and ChEMBL databases...")
    (
        ligs_smiles_merged,
        binding_data_merged,
        uncurated_binding_data,
    ) = merge_databases(
        data_dir,
        tanimoto_curation_threshold=args.tanimoto_curation_threshold,
    )

    merged_dir = os.path.join(output_dir, "merged_databases")
    os.makedirs(merged_dir, exist_ok=True)

    ligs_smiles_merged_path = os.path.join(
        merged_dir, "ligs_smiles_merged.parquet"
    )
    binding_data_merged_path = os.path.join(
        merged_dir, "binding_data_merged.parquet"
    )
    uncurated_binding_data_path = os.path.join(
        merged_dir, "uncurated_binding_data.parquet"
    )

    print(f"[INFO] Saving merged ligands to {ligs_smiles_merged_path}")
    ligs_smiles_merged.to_parquet(ligs_smiles_merged_path, index=False)

    print(f"[INFO] Saving merged binding data to {binding_data_merged_path}")
    binding_data_merged.to_parquet(binding_data_merged_path, index=False)

    print(f"[INFO] Saving uncurated binding data to {uncurated_binding_data_path}")
    uncurated_binding_data.to_parquet(uncurated_binding_data_path, index=False)

    # ------------------------------------------------------------------
    # 6) UniProt: update pickle and export FASTA
    # ------------------------------------------------------------------
    sequences_dir = os.path.join(output_dir, "sequences")
    os.makedirs(sequences_dir, exist_ok=True)

    print("[INFO] Updating UniProt target sequences pickle...")
    uniprot_sequences = update_target_sequences_pickle(
        binding_data_merged,
        output_dir=sequences_dir,
    )

    print("[INFO] Exporting UniProt sequences to FASTA...")
    uniprot_dict_to_fasta(output_dir=sequences_dir)

    metadata["uniprot_last_update"] = date.today().isoformat()

    # ------------------------------------------------------------------
    # 7) Save updated metadata
    # ------------------------------------------------------------------
    print(f"[INFO] Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print("[INFO] Full pipeline finished successfully.")


if __name__ == "__main__":
    main()
