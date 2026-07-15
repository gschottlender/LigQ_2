export type SearchType = 'sequence' | 'nearest_k' | 'domain';
export type RankingSource = 'blast' | 'hmmer';
export type LigandSource = 'pdb' | 'chembl';
export type SearchState = 'idle' | 'running' | 'done';
export type JobStatus =
  | 'queued'
  | 'running'
  | 'partial_results'
  | 'completed'
  | 'completed_with_warnings'
  | 'failed'
  | 'cancelled'
  | 'interrupted';

export interface SearchResultsSummary {
  qseqid: string;
  n_proteins_sequence: number;
  n_proteins_nearest_k: number;
  n_proteins_domain: number;
  n_known_ligands_sequence: number;
  n_known_ligands_nearest_k: number;
  n_known_ligands_domain: number;
  n_predicted_ligands_sequence: number;
  n_predicted_ligands_nearest_k: number;
  n_predicted_ligands_domain: number;
}

export interface ProteinRanking {
  protein_rank: number;
  qseqid: string;
  sseqid: string;
  search_type: SearchType;
  ranking_source: RankingSource;
  blast_bitscore: number | null;
  blast_evalue: number | null;
  blast_pident: number | null;
  blast_qcov: number | null;
  blast_scov: number | null;
  best_domain_score: number | null;
  best_domain_evalue: number | null;
  n_shared_domains: number;
}

export interface KnownLigand {
  search_type: SearchType;
  uniprot_id: string;
  chem_comp_id: string;
  source: LigandSource;
  binding_sites: string[];
  pdb_ids: string[];
  pchembl: number | null;
  mechanism: string | null;
  activity_comment: string | null;
  curation_method: string | null;
  smiles: string;
}

export interface PredictedLigand {
  search_type: SearchType;
  uniprot_id: string;
  chem_comp_id: string;
  query_id: string;
  tanimoto?: number | null;
  similarity?: number | null;
  bsi_score?: number | null;
  smiles: string;
  qseqid: string;
  sseqid: string;
}

export interface QueryResult {
  summary: SearchResultsSummary;
  proteins?: ProteinRanking[];
  knownLigands?: KnownLigand[];
  predictedLigands?: PredictedLigand[];
  status: JobStatus;
  errorMessage?: string;
  warningMessage?: string;
  progressPercent?: number;
}

export interface Database {
  id: string;
  label: string;
}

export interface RepresentationOption {
  id: string;
  label: string;
  metric: 'tanimoto' | 'cosine';
  databaseId: string;
  defaultThreshold: number | null;
}

export interface JobProgress {
  step: string;
  label: string;
  step_index: number;
  step_count: number;
  percent: number;
  current: number | null;
  total: number | null;
  unit: string | null;
  context: string | null;
  eta_seconds: number | null;
  downloaded_bytes: number | null;
  download_total_bytes: number | null;
  completed_files: number | null;
  total_files: number | null;
}

export interface JobFailure {
  step: string | null;
  label: string;
  step_index: number | null;
  step_count: number | null;
  message: string;
}

export interface Job {
  job_id: string;
  job_type: 'setup' | 'search' | 'build_database' | 'add_representation';
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed_seconds: number | null;
  progress_message: string;
  progress_percent: number | null;
  progress: JobProgress | null;
  output_dir: string | null;
  warnings: string[];
  error: string | null;
  failure: JobFailure | null;
  completed_queries: string[];
  all_queries: string[];
  n_queries: number | null;
}

export interface SetupStatus {
  ready: boolean;
  state: 'ready' | 'required' | 'downloading';
  repo_id: string;
  revision: string;
  required_download_bytes: number;
  total_required_bytes: number;
  available_bytes: number;
  enough_space: boolean;
  required_file_count: number;
  total_file_count: number;
  missing_paths: string[];
  size_source: 'huggingface' | 'repository_snapshot';
  metadata_error: string | null;
  job_id: string | null;
  job_status: JobStatus | null;
}
