import React from 'react';

const base = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round' };

export const IconProfile = (p) => (
  <svg width="15" height="15" viewBox="0 0 24 24" {...base} {...p}>
    <circle cx="9" cy="8" r="3.2" /><path d="M3 20c.6-3.4 3-5.2 6-5.2s5.4 1.8 6 5.2" /><path d="M17 9h4M19 7v4" />
  </svg>
);
export const IconRisk = (p) => (
  <svg width="15" height="15" viewBox="0 0 24 24" {...base} {...p}>
    <path d="M3 17l5-6 4 3 5-8 4 5" /><path d="M3 21h18" />
  </svg>
);
export const IconBook = (p) => (
  <svg width="15" height="15" viewBox="0 0 24 24" {...base} {...p}>
    <path d="M4 5.5A1.5 1.5 0 015.5 4H11v16H5.5A1.5 1.5 0 014 18.5z" /><path d="M20 5.5A1.5 1.5 0 0018.5 4H13v16h5.5A1.5 1.5 0 0020 18.5z" />
  </svg>
);
export const IconSearch = (p) => (
  <svg width="15" height="15" viewBox="0 0 24 24" {...base} {...p}>
    <circle cx="11" cy="11" r="6.5" /><path d="M16 16l4.5 4.5" />
  </svg>
);
export const IconGraph = (p) => (
  <svg width="15" height="15" viewBox="0 0 24 24" {...base} {...p}>
    <circle cx="6" cy="6" r="2.4" /><circle cx="18" cy="8" r="2.4" /><circle cx="12" cy="18" r="2.4" />
    <path d="M8 7l8 1M7 8l4 8M16 10l-3 6" />
  </svg>
);
