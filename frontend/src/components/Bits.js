import React from 'react';
import theme from '../theme';

export function PanelHead({ title, note, right }) {
  return (
    <div className="panel-head" style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
      <div style={{ minWidth: 0 }}>
        <h3>{title}</h3>
        {note ? <p>{note}</p> : null}
      </div>
      {right ? <div style={{ marginLeft: 'auto' }}>{right}</div> : null}
    </div>
  );
}

export function Meter({ value, color = theme.accent, width }) {
  const v = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <span className="meter" style={width ? { width } : undefined}>
      <i style={{ width: `${v}%`, background: color }} />
    </span>
  );
}

export function Figures({ items }) {
  return (
    <section className="figures">
      {items.map((it) => (
        <div className="figure" key={it.label}>
          <b>{it.value}</b>
          <span>{it.label}</span>
        </div>
      ))}
    </section>
  );
}

export function NoStudent({ title, body, onPick, samples = ['ST1008', 'ST1024', 'ST1047'] }) {
  return (
    <section className="empty">
      <h3>{title}</h3>
      <p>{body}</p>
      <div className="try">
        <span className="hint">Try:</span>
        {samples.map((s) => (
          <button key={s} type="button" onClick={() => onPick && onPick(s)}>{s}</button>
        ))}
      </div>
    </section>
  );
}

export function ChartTip({ active, payload, label, unit = '' }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tip">
      {label !== undefined && label !== null ? <b>{label}</b> : null}
      {payload.map((p) => (
        <div key={p.dataKey || p.name} style={{ color: p.color || theme.ink }}>
          {p.name}: <span className="mono">{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}{unit}</span>
        </div>
      ))}
    </div>
  );
}

export function KeyValue({ items }) {
  return (
    <div>
      {items.map((i) => (
        <div className="kv" key={i.k}>
          <span className="muted">{i.k}</span>
          <b>{i.v}</b>
        </div>
      ))}
    </div>
  );
}

export function Loading({ text = 'Loading trained output...' }) {
  return <div className="loading">{text}</div>;
}

export function ErrorBox({ message }) {
  if (!message) return null;
  return <div className="error-box">{message}</div>;
}
