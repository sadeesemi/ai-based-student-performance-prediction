import React, { useEffect, useRef, useState } from 'react';
import { searchStudents } from '../services/api';
import { useStudent } from '../StudentContext';
import { IconSearch } from './Icons';

export default function StudentSearch({ placeholder = 'Search student ID or name, e.g. ST1008' }) {
  const { query, setQuery, selectStudent, clearStudent } = useStudent();
  const [items, setItems] = useState([]);
  const [open, setOpen] = useState(false);
  const box = useRef(null);

  useEffect(() => {
    let alive = true;
    if (!query || query.length < 2) {
      setItems([]);
      return () => { alive = false; };
    }
    searchStudents(query).then((r) => { if (alive) setItems(r); }).catch(() => {});
    return () => { alive = false; };
  }, [query]);

  useEffect(() => {
    const onDoc = (e) => { if (box.current && !box.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  const pick = (id) => {
    setOpen(false);
    selectStudent(id);
  };

  return (
    <div className="topbar-search" ref={box}>
      <form
        className="search-field"
        onSubmit={(e) => {
          e.preventDefault();
          const first = items[0];
          pick(first ? first.id : query);
        }}
      >
        <IconSearch />
        <input
          value={query}
          placeholder={placeholder}
          onFocus={() => setOpen(true)}
          onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        />
        {query ? (
          <button type="button" className="search-clear" onClick={clearStudent} title="Clear">&times;</button>
        ) : null}
      </form>
      {open && items.length > 0 ? (
        <div className="suggest">
          {items.map((s) => (
            <button key={s.id} type="button" onClick={() => pick(s.id)}>
              <div className="s-id">{s.id}</div>
              <div className="s-name">{s.name}</div>
              <div className="s-meta">{s.program} &middot; {s.segment} &middot; {s.riskLevel}</div>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
