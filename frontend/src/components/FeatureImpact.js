import React from 'react';
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import theme from '../theme';
import { ChartTip } from './Bits';

/**
 * Local explanation chart - additive tree contributions (SHAP-style) for the
 * learner's top ranked resource. Positive bars pushed the relevance score up.
 */
export default function FeatureImpact({ data, height = 250 }) {
  if (!data || !data.length) return <p className="muted">No explanation available.</p>;
  const rows = data.map((d) => ({
    label: d.label || d.feature,
    impact: Number(d.impact.toFixed(5))
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 18, bottom: 4, left: 4 }}>
        <XAxis type="number" tick={{ fontSize: 10, fill: theme.inkSoft }} axisLine={false} tickLine={false} />
        <YAxis
          type="category"
          dataKey="label"
          width={168}
          tick={{ fontSize: 10, fill: theme.ink }}
          axisLine={false}
          tickLine={false}
        />
        <ReferenceLine x={0} stroke={theme.line} />
        <Tooltip content={<ChartTip />} />
        <Bar dataKey="impact" name="contribution" radius={[0, 3, 3, 0]} isAnimationActive={false} barSize={11}>
          {rows.map((r) => (
            <Cell key={r.label} fill={r.impact >= 0 ? theme.accent : theme.high} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
