import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { IconBook, IconGraph, IconProfile, IconRisk } from './Icons';
import { openKnowledgeGraph, openVisualReport } from '../services/api';
import { useStudent } from '../StudentContext';

const LINKS = [
  { to: '/profiling', label: 'Profiling & Segmentation', no: '01', Icon: IconProfile },
  { to: '/prediction', label: 'Performance & Risk', no: '02', Icon: IconRisk },
  { to: '/recommendations', label: 'Recommendations', no: '03', Icon: IconBook }
];

export default function Sidebar() {
  const navigate = useNavigate();
  const { student, meta } = useStudent();
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">AI</div>
        <div>
          <div className="brand-name">Student Performance</div>
          <div className="brand-sub">Prediction System</div>
        </div>
      </div>

      <div className="side-label">Modules</div>
      {LINKS.map(({ to, label, no, Icon }) => (
        <NavLink key={to} to={to} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
          <Icon />
          <span>{label}</span>
          <span className="no">{no}</span>
        </NavLink>
      ))}

    
      <div className="side-foot">
      
        <button className="side-btn" style={{ marginTop: 10 }} type="button" onClick={() => navigate('/login')}>
          Sign out
        </button>
      </div>
    </aside>
  );
}
