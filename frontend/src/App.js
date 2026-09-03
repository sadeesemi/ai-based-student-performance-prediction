import React from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import Login from './pages/Login';
import Module1Profiling from './pages/Module1Profiling';
import Module2Prediction from './pages/Module2Prediction';
import Module3Recommendations from './pages/Module3Recommendations';

export const isAuthed = () => localStorage.getItem('m3_auth') === '1';

function Protected({ children }) {
  const location = useLocation();
  if (!isAuthed()) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return children;
}

function Shell({ eyebrow, title, children }) {
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main">
        <TopBar eyebrow={eyebrow} title={title} />
        <div className="content">{children}</div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/profiling" element={(
        <Protected>
          <Shell eyebrow="Module 01" title="Student Profiling &amp; Segmentation">
            <Module1Profiling />
          </Shell>
        </Protected>
      )} />
      <Route path="/prediction" element={(
        <Protected>
          <Shell eyebrow="Module 02" title="Performance &amp; Risk Prediction">
            <Module2Prediction />
          </Shell>
        </Protected>
      )} />
      <Route path="/recommendations" element={(
        <Protected>
          <Shell eyebrow="Module 03" title="Personalized Intervention Recommendations">
            <Module3Recommendations />
          </Shell>
        </Protected>
      )} />
      <Route path="*" element={<Navigate to={isAuthed() ? '/recommendations' : '/login'} replace />} />
    </Routes>
  );
}
