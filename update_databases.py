import os

from generate_databases.pdb_db import generate_pdb_database, update_pdb_database_from_dir
from generate_databases.chembl_db import generate_chembl_database
from generate_databases.uniprot_db import update_target_sequences_pickle
from compound_processing.compound_helpers import (
    unify_pdb_chembl,
    build_ligand_index,
    build_morgan_representation,
    LigandStore,
)
from huggingface_hub import HfApi, hf_hub_download


def parse_args():
    # output_dir_default = 'databases'
    return args

def download_base_databases_from_huggingface(output_dir):
    repo_id = "gschottlender/LigQ_2"  # <-- cambiá esto
    local_dir = output_dir  # destino local

    api = HfApi()

    # Crear carpeta local si no existe
    os.makedirs(local_dir, exist_ok=True)

    # Listar todos los archivos del repo
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Descargar los archivos

def main(args):
    args = parse_args()

    # Open metadata
    
    # If no local database, download preprocessed pdb database from huggingface repo and update
    # Si no existe el output_dir:
        download_base_databases_from_huggingface(output_dir)
    
    metadata = json.load(f'{output_dir}/db_metadata.json')
    # Update pdb database
    pdb_db_dir =f'{output_dir}/pdb'
    update_pdb_database_from_dir(pdb_db_dir,temp_dir=args.temp_data_dir)

    # Update chembl database
    if metadata['chembl_version'] > args.chembl_version:
        chembl_sql_dir = f"{args.temp_data_dir}/chembl_sql"
        os.makedirs(chembl_sql_dir, exist_ok=True)
        chembl_file = f"{chembl_sql_dir}/chembl_{chembl_version}_sqlite.tar.gz"

        # Delete old .gz file (from truncated downloads for example)
        if os.path.exists(chembl_file):
            os.remove(chembl_file)

        # Download ChEMBL SQLite
        url = (
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
            f"chembl_{chembl_version}/chembl_{chembl_version}_sqlite.tar.gz"
        )

        os.system(f"wget -q -P {chembl_sql_dir} {url}")

        # Extract .tar.gz
        os.system(f"tar -xvzf {chembl_file} -C {chembl_sql_dir}")

        # Find the .db file
        db_filename = f"chembl_{chembl_version}.db"
        chembl_db_path = None

        for root, dirs, files in os.walk(chembl_sql_dir):
            if db_filename in files:
                chembl_db_path = os.path.join(root, db_filename)
                break
        
        generate_chembl_database(
            chembl_db_path=chembl_db_path,
            output_dir=f"{data_dir}/chembl"
        )
    
    # Merge databases
    ligs_smiles_merged, binding_data_merged, uncurated_binding_data = merge_databases(data_dir,tanimoto_curation_threshold=args.tanimoto_curation_threshold) #default 0.35
    # Obtain uniprot_sequences
    uniprot_sequences = update_target_sequences_pickle(binding_data_merged)