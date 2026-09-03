import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import api from './services/api';

const StudentContext = createContext(null);

export function StudentProvider({ children }) {
  const [meta, setMeta] = useState(null);
  const [metaError, setMetaError] = useState('');
  const [student, setStudent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');

  useEffect(() => {
    api.getMeta()
      .then(setMeta)
      .catch((e) => setMetaError(e.message));
    api.getIndex().catch(() => {});
  }, []);

  const selectStudent = useCallback(async (id) => {
    setLoading(true);
    setError('');
    try {
      const s = await api.getStudent(id);
      setStudent(s);
      setQuery(s.id);
      return s;
    } catch (e) {
      setStudent(null);
      setError(e.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearStudent = useCallback(() => {
    setStudent(null);
    setError('');
    setQuery('');
  }, []);

  const value = useMemo(
    () => ({ meta, metaError, student, loading, error, query, setQuery, selectStudent, clearStudent }),
    [meta, metaError, student, loading, error, query, selectStudent, clearStudent]
  );

  return <StudentContext.Provider value={value}>{children}</StudentContext.Provider>;
}

export function useStudent() {
  const ctx = useContext(StudentContext);
  if (!ctx) throw new Error('useStudent must be used inside <StudentProvider>');
  return ctx;
}

export default StudentContext;
