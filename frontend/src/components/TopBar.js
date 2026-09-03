import React from 'react';
import StudentSearch from './StudentSearch';

export default function TopBar({ eyebrow, title }) {
  return (
    <header className="topbar">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
      </div>
      <StudentSearch />
    </header>
  );
}
