import { useState, useRef, useCallback, useEffect } from 'react';
import { ChevronDown, ChevronRight, FileUp, FolderOpen, Info, Loader2, Settings, Upload, X } from 'lucide-react';
import { useDatabase } from '../../context/DatabaseContext';
import { api } from '../../lib/api';
import { JobFailurePanel, JobProgressPanel } from '../../components/JobProgressPanel';
import { useJobPolling } from '../../hooks/useJobPolling';

type FileExt = 'smi' | 'csv' | 'tsv' | 'parquet';

const SMILES_LINE_RE = /^[A-Za-z0-9()[\]=#$+\-@/\\%.:\s*]+$/;

function validateSmi(text: string): string | null {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean).slice(0, 10);
  if (lines.length === 0) return 'File does not appear to be a valid SMILES file.';
  const invalid = lines.filter((l) => !SMILES_LINE_RE.test(l));
  if (invalid.length > lines.length / 2) return 'File does not appear to be a valid SMILES file.';
  return null;
}

function validateCsvTsv(text: string, sep: string): string | null {
  const firstLine = text.split('\n').map((l) => l.trim()).find(Boolean) ?? '';
  if (firstLine.split(sep).length < 2) {
    return 'File must have at least 2 columns (compound ID and SMILES).';
  }
  return null;
}

interface ProcessingState {
  stage: 'idle' | 'processing' | 'done' | 'error';
  message: string;
}

const ID_CANDIDATES = ['id', 'zinc_id', 'compound_id'];
const SMILES_CANDIDATES = ['smiles', 'SMILES', 'canonical_smiles'];

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function toSafeName(stem: string): string {
  return stem.toLowerCase().replace(/[^a-zA-Z0-9_]/g, '_').replace(/^_+|_+$/g, '');
}

function autoSelectId(cols: string[]): string {
  return cols.find((c) => ID_CANDIDATES.includes(c)) ??
    cols.find((c) => ID_CANDIDATES.includes(c.toLowerCase())) ?? '';
}

function autoSelectSmiles(cols: string[]): string {
  return cols.find((c) => SMILES_CANDIDATES.includes(c)) ??
    cols.find((c) => c.toLowerCase() === 'smiles') ?? '';
}

function ColumnSelect({
  label, value, onChange, columns, error,
}: {
  label: string; value: string; onChange: (v: string) => void; columns: string[]; error?: string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium text-gray-600 dark:text-gray-300">{label}</label>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 pr-8 text-sm
            text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 cursor-pointer focus:outline-none focus:ring-1 focus:ring-teal-500"
        >
          <option value="">Select column…</option>
          {columns.map((col) => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>
        <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
      </div>
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  );
}

export function AddNewDatabase() {
  const { refetchDatabases } = useDatabase();

  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [fileExt, setFileExt] = useState<FileExt | null>(null);
  const [detectedColumns, setDetectedColumns] = useState<string[]>([]);
  const [compoundIdCol, setCompoundIdCol] = useState('');
  const [smilesCol, setSmilesCol] = useState('');
  const [outputName, setOutputName] = useState('');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [processing, setProcessing] = useState<ProcessingState>({ stage: 'idle', message: '' });
  const [jobId, setJobId] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);
  const [fileError, setFileError] = useState('');
  const [fileInfo, setFileInfo] = useState('');
  const [fileValidating, setFileValidating] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileValidationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (fileValidationTimerRef.current) clearTimeout(fileValidationTimerRef.current);
    };
  }, []);

  const handleJobCompleted = useCallback(async () => {
    setProcessing({ stage: 'done', message: '' });
    await refetchDatabases();
  }, [refetchDatabases]);

  const handleJobFailed = useCallback((job: { error: string | null }) => {
    setProcessing({ stage: 'error', message: job.error ?? 'Processing failed.' });
  }, []);

  const { job, resetJob } = useJobPolling(jobId, {
    onCompleted: handleJobCompleted,
    onFailed: handleJobFailed,
  });

  const applyColumns = useCallback((cols: string[]) => {
    setDetectedColumns(cols);
    setCompoundIdCol(autoSelectId(cols));
    setSmilesCol(autoSelectSmiles(cols));
  }, []);

  const parseFile = useCallback(async (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase() as FileExt | undefined;
    const safeExt = (['smi', 'csv', 'tsv', 'parquet'] as FileExt[]).includes(ext as FileExt)
      ? (ext as FileExt)
      : null;
    setFileExt(safeExt);
    setCompoundIdCol('');
    setSmilesCol('');
    setDetectedColumns([]);
    setOutputName(toSafeName(file.name.replace(/\.[^.]+$/, '')));

    if (safeExt === 'csv' || safeExt === 'tsv') {
      const sep = safeExt === 'tsv' ? '\t' : ',';
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const firstLine = text.split('\n')[0] ?? '';
        const cols = firstLine.split(sep).map((c) => c.trim().replace(/^"|"$/g, '')).filter(Boolean);
        applyColumns(cols);
      };
      reader.readAsText(file.slice(0, 8192));
    } else if (safeExt === 'parquet') {
      try {
        const formData = new FormData();
        formData.append('file', file);
        const { data } = await api.post<{ columns: string[] }>('/files/upload', formData);
        applyColumns(data.columns.length > 0 ? data.columns : ['compound_id', 'smiles']);
      } catch {
        applyColumns(['compound_id', 'smiles']);
      }
    }
  }, [applyColumns]);

  const acceptFile = useCallback((file: File) => {
    setFileError('');
    setFileInfo('');
    setErrors((prev) => ({ ...prev, file: '' }));

    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['smi', 'csv', 'tsv', 'parquet'].includes(ext ?? '')) {
      setFileError('Invalid file type. Accepted: .smi, .csv, .tsv, .parquet');
      return;
    }

    if (ext === 'parquet') {
      setFileInfo('Parquet file selected. Column mapping will be available after upload.');
      setUploadedFile(file);
      parseFile(file);
      return;
    }

    if (fileValidationTimerRef.current) clearTimeout(fileValidationTimerRef.current);
    fileValidationTimerRef.current = setTimeout(() => setFileValidating(true), 200);

    const reader = new FileReader();
    reader.onload = (ev) => {
      if (fileValidationTimerRef.current) clearTimeout(fileValidationTimerRef.current);
      setFileValidating(false);
      const text = ev.target?.result as string;
      let error: string | null = null;
      if (ext === 'smi') error = validateSmi(text);
      else if (ext === 'csv') error = validateCsvTsv(text, ',');
      else if (ext === 'tsv') error = validateCsvTsv(text, '\t');
      if (error) {
        setFileError(error);
        return;
      }
      setUploadedFile(file);
      parseFile(file);
    };
    reader.onerror = () => {
      if (fileValidationTimerRef.current) clearTimeout(fileValidationTimerRef.current);
      setFileValidating(false);
      setFileError('Could not read the file.');
    };
    reader.readAsText(file.slice(0, 8192));
  }, [parseFile]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) acceptFile(file);
  }, [acceptFile]);

  const handleBrowse = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) acceptFile(file);
  };

  const validate = () => {
    const errs: Record<string, string> = {};
    if (!uploadedFile) errs.file = 'Please upload a file.';
    if (!outputName.trim()) {
      errs.outputName = 'Database name cannot be empty.';
    } else if (!/^[a-zA-Z0-9_]+$/.test(outputName.trim())) {
      errs.outputName = 'Only letters, digits and underscores are allowed.';
    }
    if (fileExt === 'csv' || fileExt === 'tsv' || fileExt === 'parquet') {
      if (!compoundIdCol) errs.compoundIdCol = 'Select the Compound ID column.';
      if (!smilesCol) errs.smilesCol = 'Select the SMILES column.';
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleProcess = async () => {
    if (!validate() || !uploadedFile) return;

    resetJob();
    setJobId(null);
    setProcessing({ stage: 'processing', message: 'Submitting job…' });

    try {
      const formData = new FormData();
      formData.append('input_file', uploadedFile);
      formData.append('base_name', outputName.trim());
      if (fileExt === 'csv' || fileExt === 'tsv' || fileExt === 'parquet') {
        formData.append('id_column', compoundIdCol);
        formData.append('smiles_column', smilesCol);
      }

      const { data } = await api.post<{ job_id: string }>('/jobs/build-database', formData);
      setJobId(data.job_id);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { status?: number; data?: { message?: string } } };
      if (axiosErr.response?.status === 409) {
        setProcessing({ stage: 'error', message: `A database named "${outputName.trim()}" already exists.` });
      } else {
        const message = axiosErr.response?.data?.message ?? 'Failed to submit job.';
        setProcessing({ stage: 'error', message });
      }
    }
  };

  const reset = () => {
    setUploadedFile(null);
    setFileExt(null);
    setDetectedColumns([]);
    setCompoundIdCol('');
    setSmilesCol('');
    setOutputName('');
    setErrors({});
    setProcessing({ stage: 'idle', message: '' });
    setJobId(null);
    resetJob();
    setFileError('');
    setFileInfo('');
    setFileValidating(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const needsMapping = fileExt === 'csv' || fileExt === 'tsv' || fileExt === 'parquet';
  const canSubmit = !!uploadedFile && !!outputName.trim() && processing.stage !== 'processing';

  return (
    <div className="w-full">
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
        Upload a compound file and process it into a searchable database.
      </p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center gap-3 transition-colors cursor-pointer
          ${isDragging
            ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20'
            : 'border-gray-300 dark:border-gray-600 hover:border-teal-400 dark:hover:border-teal-500'
          }`}
      >
        <FileUp className={`w-10 h-10 ${isDragging ? 'text-teal-500' : 'text-gray-400'}`} />
        {uploadedFile ? (
          <div className="flex flex-col items-center gap-1.5">
            <div className="flex items-center gap-2">
              <span className="px-1.5 py-0.5 text-xs font-mono font-semibold bg-teal-100 dark:bg-teal-900/40 text-teal-700 dark:text-teal-300 rounded uppercase">
                {fileExt}
              </span>
              <span className="text-sm font-medium text-teal-700 dark:text-teal-300 truncate max-w-xs">
                {uploadedFile.name}
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); reset(); }}
                className="text-gray-400 hover:text-red-500 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <span className="text-xs text-gray-400 dark:text-gray-500">{formatFileSize(uploadedFile.size)}</span>
          </div>
        ) : (
          <>
            <p className="text-sm text-gray-600 dark:text-gray-300 text-center">
              Drag and drop your file here, or{' '}
              <span className="text-teal-600 dark:text-teal-400 font-medium">browse files</span>
            </p>
            <p className="text-xs text-gray-400">Accepted: .smi · .csv · .tsv · .parquet</p>
          </>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
          className="flex items-center gap-2 border border-gray-300 dark:border-gray-500 text-gray-600 dark:text-gray-300 cursor-pointer
            px-4 py-1.5 rounded-lg text-sm hover:border-teal-500 hover:text-teal-600 transition-colors"
        >
          <FolderOpen className="w-4 h-4" /> Browse files
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".smi,.csv,.tsv,.parquet"
          className="hidden"
          onChange={handleBrowse}
        />
      </div>
      {errors.file && <p className="text-xs text-red-500 mt-1">{errors.file}</p>}
      {fileValidating && (
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">Validating…</p>
      )}
      {fileError && !fileValidating && (
        <p className="text-xs text-red-500 mt-1">{fileError}</p>
      )}
      {fileInfo && !fileValidating && !fileError && (
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{fileInfo}</p>
      )}

      {/* Column mapping (csv / tsv / parquet only) */}
      {needsMapping && detectedColumns.length > 0 && (
        <div className="mt-5 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">Column mapping</p>
          <div className="flex flex-col gap-3">
            <ColumnSelect
              label="Compound ID column"
              value={compoundIdCol}
              onChange={setCompoundIdCol}
              columns={detectedColumns}
              error={errors.compoundIdCol}
            />
            <ColumnSelect
              label="SMILES column"
              value={smilesCol}
              onChange={setSmilesCol}
              columns={detectedColumns}
              error={errors.smilesCol}
            />
          </div>
        </div>
      )}

      {/* Output name */}
      <div className="mt-5 flex flex-col gap-1.5">
        <label className="text-sm font-medium text-gray-600 dark:text-gray-300">
          Output directory name
        </label>
        <input
          type="text"
          value={outputName}
          onChange={(e) => setOutputName(e.target.value)}
          placeholder="e.g. vendor_library"
          className="border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm
            text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-teal-500
            placeholder:text-gray-400"
        />
        <p className="text-xs text-gray-400 dark:text-gray-500">
          Lowercase letters, digits and underscores work best.
        </p>
        {errors.outputName && <p className="text-xs text-red-500">{errors.outputName}</p>}
      </div>

      {/* Process button */}
      <button
        onClick={handleProcess}
        disabled={!canSubmit}
        className={`mt-6 w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium transition-colors
          ${canSubmit
            ? 'bg-[#0d5c6b] hover:bg-[#0a4d5a] text-white cursor-pointer'
            : 'bg-gray-100 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
          }`}
      >
        {processing.stage === 'processing'
          ? <><Loader2 className="w-4 h-4 animate-spin" /> Processing…</>
          : <><Settings className="w-4 h-4" /> Process database</>
        }
      </button>

      {processing.stage === 'processing' && (
        <JobProgressPanel
          progress={job?.progress}
          fallbackPercent={job?.progress_percent ?? 0}
          fallbackMessage={job?.progress_message || processing.message}
          startedAt={job?.started_at}
        />
      )}

      {/* Error */}
      {processing.stage === 'error' && (
        <JobFailurePanel failure={job?.failure} error={processing.message} />
      )}

      {/* Success */}
      {processing.stage === 'done' && (
        <div className="mt-4 flex items-center gap-2 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-700 rounded-xl">
          <Upload className="w-4 h-4 text-green-600 dark:text-green-400 shrink-0" />
          <p className="text-sm text-green-700 dark:text-green-300">
            Database <span className="font-medium">"{outputName}"</span> processed. It is now available in Search.
          </p>
        </div>
      )}

      {/* What happens during processing? */}
      <div className="mt-6 border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
        <button
          onClick={() => setShowInfo((v) => !v)}
          className="w-full flex items-center gap-2 px-4 py-3 text-sm font-medium text-gray-600 dark:text-gray-300 cursor-pointer
            hover:bg-gray-50 dark:hover:bg-gray-800/40 transition-colors text-left"
        >
          <Info className="w-4 h-4 text-gray-400 shrink-0" />
          <span className="flex-1">What happens during processing?</span>
          {showInfo
            ? <ChevronDown className="w-4 h-4 text-gray-400" />
            : <ChevronRight className="w-4 h-4 text-gray-400" />
          }
        </button>
        {showInfo && (
          <ul className="px-4 pb-4 pt-1 space-y-2 text-xs text-gray-500 dark:text-gray-400 border-t border-gray-100 dark:border-gray-700/60">
            <li className="flex gap-2"><span className="text-teal-500 shrink-0">•</span>SMILES strings are canonicalized and de-duplicated.</li>
            <li className="flex gap-2"><span className="text-teal-500 shrink-0">•</span>Default Morgan fingerprints (1024 bits, radius 2) are computed.</li>
            <li className="flex gap-2"><span className="text-teal-500 shrink-0">•</span>Compounds with invalid SMILES are skipped and reported.</li>
            <li className="flex gap-2"><span className="text-teal-500 shrink-0">•</span>You can add other representations later from the next tab.</li>
          </ul>
        )}
      </div>
    </div>
  );
}
