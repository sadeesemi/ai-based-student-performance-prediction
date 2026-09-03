import React from 'react';
import theme from '../theme';

const PRIORITY_COLOR = {
  Critical: theme.high,
  High: '#e07a1b',
  Medium: theme.medium,
  Low: theme.low
};

export default function RiskBadge({ level, size = 'md' }) {
  const color = theme.riskColor[level] || theme.inkSoft;
  return (
    <span
      className="badge"
      style={{
        background: `${color}1f`,
        color,
        fontSize: size === 'sm' ? 10.5 : 11.5,
        padding: size === 'sm' ? '3px 9px' : '5px 12px'
      }}
    >
      <span className="dot" style={{ background: color }} />
      {level}
    </span>
  );
}

export function PriorityBadge({ priority }) {
  const color = PRIORITY_COLOR[priority] || theme.inkSoft;
  return (
    <span className="badge" style={{ background: `${color}1c`, color }}>
      <span className="dot" style={{ background: color }} />
      {priority}
    </span>
  );
}

export function SegmentBadge({ segment }) {
  const color = theme.segmentColor[segment] || theme.accent;
  return (
    <span className="badge" style={{ background: `${color}18`, color }}>
      <span className="dot" style={{ background: color }} />
      {segment}
    </span>
  );
}
