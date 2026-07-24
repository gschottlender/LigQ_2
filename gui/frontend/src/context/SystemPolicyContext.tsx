/* eslint-disable react-refresh/only-export-components */
import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import { api } from '../lib/api';
import type { SystemPolicy } from '../types';

interface SystemPolicyContextValue {
  policy: SystemPolicy;
  isWeb: boolean;
  reloadPolicy: () => Promise<void>;
}

const SystemPolicyContext = createContext<SystemPolicyContextValue | null>(null);

export function SystemPolicyProvider({ children }: { children: ReactNode }) {
  const [policy, setPolicy] = useState<SystemPolicy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const reloadPolicy = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await api.get<SystemPolicy>('/system/policy');
      setPolicy(data);
    } catch {
      setError('LigQ 2 could not load the server deployment policy.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Initial API synchronization intentionally owns the policy loading state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    reloadPolicy();
  }, [reloadPolicy]);

  if (!policy) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-gray-50 px-4 dark:bg-[#111827]">
        <div className="w-full max-w-md rounded-2xl border border-gray-200 bg-white p-7 text-center shadow-sm dark:border-gray-700 dark:bg-[#1a2330]">
          {loading ? (
            <Loader2 className="mx-auto h-8 w-8 animate-spin text-cyan-800 dark:text-teal-300" />
          ) : (
            <AlertTriangle className="mx-auto h-8 w-8 text-amber-600 dark:text-amber-400" />
          )}
          <h1 className="mt-4 text-xl font-semibold text-gray-800 dark:text-gray-100">
            {loading ? 'Connecting to LigQ 2' : 'Configuration unavailable'}
          </h1>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            {error ?? 'Loading the server configuration…'}
          </p>
          {!loading && (
            <button
              type="button"
              onClick={reloadPolicy}
              className="mt-5 inline-flex cursor-pointer items-center gap-2 rounded-xl bg-cyan-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-cyan-800"
            >
              <RefreshCw className="h-4 w-4" />
              Try again
            </button>
          )}
        </div>
      </main>
    );
  }

  return (
    <SystemPolicyContext.Provider
      value={{ policy, isWeb: policy.mode === 'web', reloadPolicy }}
    >
      {children}
    </SystemPolicyContext.Provider>
  );
}

export function useSystemPolicy() {
  const context = useContext(SystemPolicyContext);
  if (!context) {
    throw new Error('useSystemPolicy must be used inside SystemPolicyProvider');
  }
  return context;
}
