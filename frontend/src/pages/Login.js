import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('student@university.edu');
  const [password, setPassword] = useState('student123');
  const [error, setError] = useState('');
  const submit = (e) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) return setError('Enter an email and password to continue.');
    setError('');
    navigate('/profiling');
  };
  return (
    <main className="login-page">
      <section className="login-intro">
        <div className="brand login-brand"><div className="brand-mark">AI</div><div><div className="brand-name">Student Performance</div><div className="brand-sub">Prediction System</div></div></div>
        <div className="login-copy">
          <span className="eyebrow">Academic intelligence workspace</span>
          <h1>See the signal before the student slips.</h1>
          <p>Profile learning behaviour, predict academic risk and connect every signal to a practical intervention.</p>
          <div className="login-points">
            <div><span>01</span><strong>Profile</strong><small>Academic, behavioural and engagement patterns</small></div>
            <div><span>02</span><strong>Predict</strong><small>Risk bands with explainable feature drivers</small></div>
            <div><span>03</span><strong>Intervene</strong><small>Resources and actions matched to need</small></div>
          </div>
        </div>
        <div className="login-footer">AI-Based Student Performance Prediction System · Front-end demonstration</div>
      </section>
      <section className="login-form-wrap">
        <form className="login-form" onSubmit={submit}>
          <div className="eyebrow">Welcome back</div><h2>Sign in to continue</h2>
          <p className="muted">Enter any demo credentials. This front end has no backend or role checks.</p>
          <label htmlFor="email">Email address</label><input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@university.edu" />
          <label htmlFor="password">Password</label><input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Enter password" />
          {error && <div className="login-error">{error}</div>}
          <button className="login-button" type="submit">Open student profile <span>↗</span></button>
          <p className="login-hint">Demo values are prefilled. Click login to enter Module 01.</p>
        </form>
      </section>
    </main>
  );
}