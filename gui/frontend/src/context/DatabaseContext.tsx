import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import type { Database, RepresentationOption } from '../types';
import { api } from '../lib/api';

interface DatabaseContextValue {
  databases: Database[];
  representations: RepresentationOption[];
  addDatabase: (db: Database) => void;
  addRepresentation: (rep: RepresentationOption) => void;
  getRepresentationsForDatabase: (databaseId: string) => RepresentationOption[];
  refetchDatabases: () => Promise<void>;
  refetchRepresentationsForDatabase: (databaseId: string) => Promise<void>;
}

const DatabaseContext = createContext<DatabaseContextValue | null>(null);

async function fetchRepsForDb(name: string): Promise<RepresentationOption[]> {
  try {
    const { data } = await api.get<{
      representations: { name: string; metric: 'tanimoto' | 'cosine'; default_threshold?: number | null }[];
    }>(
      `/databases/${name}/representations`,
    );
    return data.representations.map((rep): RepresentationOption => ({
      id: rep.name,
      label: rep.name,
      metric: rep.metric,
      databaseId: name,
      defaultThreshold: rep.default_threshold ?? null,
    }));
  } catch {
    return [];
  }
}

export function DatabaseProvider({ children }: { children: ReactNode }) {
  const [databases, setDatabases] = useState<Database[]>([]);
  const [representations, setRepresentations] = useState<RepresentationOption[]>([]);

  const loadAll = useCallback(async () => {
    try {
      const { data } = await api.get<{ databases: string[] }>('/databases');
      const dbs: Database[] = data.databases.map((name) => ({ id: name, label: name.toUpperCase() }));
      setDatabases(dbs);
      const repResults = await Promise.all(data.databases.map(fetchRepsForDb));
      setRepresentations(repResults.flat());
    } catch (err) {
      console.error('Failed to load databases from API:', err);
    }
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  const refetchDatabases = useCallback(async () => {
    await loadAll();
  }, [loadAll]);

  const refetchRepresentationsForDatabase = useCallback(async (databaseId: string) => {
    const reps = await fetchRepsForDb(databaseId);
    setRepresentations((prev) => [
      ...prev.filter((r) => r.databaseId !== databaseId),
      ...reps,
    ]);
  }, []);

  const addDatabase = (db: Database) => setDatabases((prev) => [...prev, db]);
  const addRepresentation = (rep: RepresentationOption) => setRepresentations((prev) => [...prev, rep]);
  const getRepresentationsForDatabase = (databaseId: string) =>
    representations.filter((r) => r.databaseId === databaseId);

  return (
    <DatabaseContext.Provider
      value={{
        databases,
        representations,
        addDatabase,
        addRepresentation,
        getRepresentationsForDatabase,
        refetchDatabases,
        refetchRepresentationsForDatabase,
      }}
    >
      {children}
    </DatabaseContext.Provider>
  );
}

export function useDatabase() {
  const ctx = useContext(DatabaseContext);
  if (!ctx) throw new Error('useDatabase must be used inside DatabaseProvider');
  return ctx;
}
