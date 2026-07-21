import { useEffect, useState } from 'react';
import { AlertTriangle, Loader2, XCircle } from 'lucide-react';
import { api } from '../lib/api';

interface CancelJobButtonProps {
  jobId: string;
  resourceLabel: string;
  description: string;
  cancelling: boolean;
  onCancelStarted: () => void;
  onCancelFinished: () => void | Promise<void>;
  onCancelError: (message: string) => void;
}

export function CancelJobButton({
  jobId,
  resourceLabel,
  description,
  cancelling,
  onCancelStarted,
  onCancelFinished,
  onCancelError,
}: CancelJobButtonProps) {
  const [confirmOpen, setConfirmOpen] = useState(false);

  useEffect(() => {
    if (!confirmOpen || cancelling) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setConfirmOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [cancelling, confirmOpen]);

  const cancelJob = async () => {
    setConfirmOpen(false);
    onCancelStarted();
    try {
      await api.delete(`/jobs/${jobId}`);
      await onCancelFinished();
    } catch (error: unknown) {
      const response = (error as {
        response?: { data?: { message?: string; detail?: { message?: string } } };
      }).response?.data;
      onCancelError(
        response?.message
          ?? response?.detail?.message
          ?? 'The job could not be cancelled. Please try again.',
      );
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setConfirmOpen(true)}
        disabled={cancelling}
        className="mt-3 flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl border border-red-300
          px-4 py-2.5 text-sm font-medium text-red-700 transition-colors hover:bg-red-50 disabled:cursor-wait
          disabled:opacity-60 dark:border-red-700 dark:text-red-300 dark:hover:bg-red-950/30"
      >
        {cancelling
          ? <><Loader2 className="h-4 w-4 animate-spin" /> Cancelling…</>
          : <><XCircle className="h-4 w-4" /> Cancel</>
        }
      </button>

      {confirmOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setConfirmOpen(false);
          }}
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby={`cancel-${jobId}-title`}
            className="w-full max-w-md rounded-2xl border border-gray-200 bg-white p-5 shadow-2xl
              dark:border-gray-700 dark:bg-gray-900"
          >
            <div className="flex items-start gap-3">
              <div className="rounded-full bg-amber-100 p-2 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div>
                <h2 id={`cancel-${jobId}-title`} className="text-base font-semibold text-gray-800 dark:text-gray-100">
                  Cancel {resourceLabel}?
                </h2>
                <p className="mt-2 text-sm leading-relaxed text-gray-600 dark:text-gray-300">
                  {description}
                </p>
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setConfirmOpen(false)}
                className="cursor-pointer rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-600
                  hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800"
              >
                Keep running
              </button>
              <button
                type="button"
                onClick={() => void cancelJob()}
                className="cursor-pointer rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
              >
                Cancel job
              </button>
            </div>
          </section>
        </div>
      )}
    </>
  );
}
