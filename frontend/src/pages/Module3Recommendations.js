import React, { useMemo } from 'react';
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';
import { ChartTip, ErrorBox, Figures, Loading, Meter, NoStudent, PanelHead } from '../components/Bits';
import FeatureImpact from '../components/FeatureImpact';
import RiskBadge, { PriorityBadge, SegmentBadge } from '../components/RiskBadge';
import { openKnowledgeGraph, openVisualReport } from '../services/api';
import { useStudent } from '../StudentContext';
import theme from '../theme';


const STRATEGY_COLORS = [theme.accent, theme.low, theme.medium, theme.inkSoft, theme.purple];

function engagementBand(score) {
  if (score >= 65) return { label: 'Strong', color: theme.low };
  if (score >= 45) return { label: 'Moderate', color: theme.medium };
  return { label: 'Weak', color: theme.high };
}

export default function Module3Recommendations() {
  const { meta, metaError, student, loading, error, selectStudent } = useStudent();

  const resourceById = useMemo(() => {
    const map = {};
    (meta ? meta.resources : []).forEach((r) => { map[r.id] = r; });
    return map;
  }, [meta]);

  if (metaError) {
    return (
      <ErrorBox message={'Could not load the trained output (' + metaError + '). Run the backend pipeline first: python main.py inside backend/recommendation_module.'} />
    );
  }
  if (!meta) return <Loading />;

  const band = student ? engagementBand(student.engagement) : null;
  const avgDuration = Math.round(meta.resources.reduce((a, r) => a + r.minutes, 0) / meta.resources.length);
  const lessonsCovered = new Set(meta.resources.map((r) => r.lesson)).size;

  return (
    <div className="stack stack-36">
      <div className="report-row">
        <button className="rbtn" type="button" onClick={() => openKnowledgeGraph(student ? student.id : null)}>
          {student ? 'Open knowledge graph for ' + student.id : 'Open interactive knowledge graph'} &rarr;
        </button>
        <button className="rbtn dark" type="button" onClick={openVisualReport}>
          Preprocessing &amp; evaluation report &rarr;
        </button>
        
      </div>

      {loading ? <Loading /> : null}
      <ErrorBox message={error} />

      {!student ? (
        <NoStudent
          title="Recommendations are per learner"
          body="Search a student ID above to see the ranked LMS resources, the actions generated from their risk drivers, and why each one was chosen. "
          onPick={selectStudent}
        />
      ) : (
        <>
          <section className="identity">
            <div className="identity-main">
              <div className="avatar">{student.name.split(' ').map((n) => n[0]).slice(0, 2).join('')}</div>
              <div>
                <div className="identity-name">{student.name}</div>
                <div className="identity-meta mono">{student.id} / {student.segment} / {student.program}</div>
              </div>
            </div>
            <div className="identity-stats">
              <div className="identity-stat"><span className="v" style={{ color: band.color }}>{band.label}</span><span className="l">Engagement level</span></div>
              <div className="identity-stat"><span className="v">{student.engagement}/100</span><span className="l">Engagement score</span></div>
              <div className="identity-stat"><span className="v">{student.recommendations.length}</span><span className="l">Resources matched</span></div>
              <div className="identity-stat"><span className="v">{student.actions.length}</span><span className="l">Actions generated</span></div>
              <div className="identity-stat"><span className="v">{student.priority.toFixed(2)}</span><span className="l">Intervention priority</span></div>
            </div>
            <SegmentBadge segment={student.segment} />
            <RiskBadge level={student.riskLevel} />
          </section>

          <section className="grid grid--wide-left">
            <div className="panel panel--flush">
              <div style={{ padding: '18px 20px 4px' }}>
                <PanelHead
                  title="Recommended for this learner"
                  note="Hybrid ranking: rule-based filter, TF-IDF content similarity, knowledge graph + GNN, random forest relevance"
                />
              </div>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Resource</th><th>Type</th><th>Signal</th><th>Reason</th>
                    <th className="num">Relevance</th><th />
                  </tr>
                </thead>
                <tbody>
                  {student.recommendations.map((rec) => {
                    const r = resourceById[rec.r] || {};
                    return (
                      <tr key={rec.r}>
                        <td>
                          <span className="res-title">{r.title}</span>
                          <span className="res-sub">
                            {r.lesson} / {r.sub} / {r.minutes} min / {r.level}
                            {rec.gap ? ' / non-engaged lesson' : ''}
                          </span>
                        </td>
                        <td className="muted">{r.type}</td>
                        <td><span className="badge badge--outline">{rec.strat}</span></td>
                        <td className="muted" style={{ maxWidth: 230 }}>{rec.why}</td>
                        <td className="num">
                          <div className="rel">
                            <Meter value={rec.rel * 100} color={theme.accent} />
                            <span>{rec.rel.toFixed(3)}</span>
                          </div>
                        </td>
                        <td>
                          <a href={r.url} target="_blank" rel="noreferrer" className="link-btn">Open</a>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="panel">
              <PanelHead title="Recommended actions" note="Derived from risk band and behaviour gaps" />
              <div className="action-list">
                {student.actions.map((a) => (
                  <div className="action" key={a.title}>
                    <div className="action-title">{a.title}</div>
                    <div className="action-detail">{a.detail}</div>
                    <div className="action-meta">
                      <PriorityBadge priority={a.priority} />
                      <span className="badge badge--plain">{a.category}</span>
                      <span className="hint" style={{ marginLeft: 'auto' }}>{a.status}</span>
                    </div>
                  </div>
                ))}
              </div>
             
            </div>
          </section>

          <section className="grid grid--wide-left">
            <div className="panel">
              <PanelHead
                title="Why these recommendations"
                note="Additive tree contributions (SHAP-style) for the top ranked resource"
                right={<span className="hint mono">bias {student.explanationBias}</span>}
              />
              <FeatureImpact data={student.features} height={260} />
            </div>
            <div className="panel">
              <PanelHead title="Coverage" note="Lessons the matcher pulled from" />
              {student.gapLessons.length === 0 ? (
                <p className="muted">No non-engaged lesson flagged, so consolidation and enrichment material was selected instead.</p>
              ) : (
                <div className="row" style={{ marginBottom: 14 }}>
                  {student.gapLessons.map((m) => <span key={m} className="badge badge--plain">{m}</span>)}
                </div>
              )}
              <div className="legend">
                {meta.topkCurve.map((q) => (
                  <div className="legend-item" key={q.k}>
                    <span className="mono" style={{ fontSize: 12 }}>@{q.k}</span>
                    <span className="muted">Precision {q.precision.toFixed(3)}</span>
                    <span className="count">Recall {q.recall.toFixed(3)}</span>
                  </div>
                ))}
              </div>
              <p className="note">
                Held-out ranking quality: NDCG@10 {meta.metrics['ndcg@10']} &middot; MAP@10 {meta.metrics['map@10']} &middot; MRR {meta.metrics.mrr}
              </p>
              <button className="rbtn ghost" type="button" style={{ marginTop: 12 }}
                onClick={() => openKnowledgeGraph(student.id)}>
                Trace this learner in the knowledge graph
              </button>
            </div>
          </section>
        </>
      )}

      
      

      <Figures
        items={[
          { label: 'Resources in library', value: meta.nResources },
          { label: 'Lessons covered', value: lessonsCovered },
          { label: 'Precision@5', value: meta.metrics['precision@5'].toFixed(3) },
          { label: 'Recall@10', value: meta.metrics['recall@10'].toFixed(3) },
          { label: 'NDCG@10', value: meta.metrics['ndcg@10'].toFixed(3) },
          { label: 'Ranker R2', value: meta.regression.r2.toFixed(3) },
          { label: 'Avg duration', value: avgDuration + ' min' }
        ]}
      />

      <section className="grid grid--wide-right">
        <div className="panel">
          <PanelHead title="Strategy mix" note="Share of ranker importance per signal family" />
          <ResponsiveContainer width="100%" height={252}>
            <PieChart>
              <Pie data={meta.strategyMix} dataKey="value" nameKey="name" innerRadius={52} outerRadius={86}
                paddingAngle={2} stroke={theme.paper} strokeWidth={2} isAnimationActive={false}>
                {meta.strategyMix.map((s, i) => (
                  <Cell key={s.name} fill={STRATEGY_COLORS[i % STRATEGY_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip content={<ChartTip unit="%" />} />
              <Legend verticalAlign="bottom" iconType="circle"
                formatter={(v) => <span style={{ fontSize: 11, color: theme.inkSoft }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="panel panel--flush">
          <div style={{ padding: '18px 20px 4px' }}>
            <PanelHead title="Resource library" note="Indexed LMS material available to the matcher" />
          </div>
          <div className="scroll-420">
            <table className="tbl">
              <thead>
                <tr><th>Title</th><th>Lesson</th><th>Type</th><th className="num">Min</th><th>Level</th></tr>
              </thead>
              <tbody>
                {meta.resources.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <a href={r.url} target="_blank" rel="noreferrer">{r.title}</a>
                      <span className="res-sub">{r.week} / {r.topic}</span>
                    </td>
                    <td className="muted">{r.lesson}</td>
                    <td className="muted">{r.type}</td>
                    <td className="num">{r.minutes}</td>
                    <td className="muted">{r.level}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="grid grid--2">
        <div className="panel panel--flush">
          <div style={{ padding: '18px 20px 4px' }}>
            <PanelHead title="Strategy ablation" note="Single-signal baselines against the proposed hybrid model" />
          </div>
          <table className="tbl">
            <thead>
              <tr><th>Strategy</th><th className="num">P@5</th><th className="num">R@10</th><th className="num">NDCG@10</th></tr>
            </thead>
            <tbody>
              {meta.ablation.map((a) => (
                <tr key={a.strategy}>
                  <td style={{ fontWeight: a.strategy.indexOf('Random Forest') >= 0 ? 700 : 400 }}>{a.strategy}</td>
                  <td className="num">{a['precision@5'].toFixed(3)}</td>
                  <td className="num">{a['recall@10'].toFixed(3)}</td>
                  <td className="num">{a['ndcg@10'].toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <PanelHead title="Ranker feature importance" note="Top drivers of the learned relevance function" />
          <div className="legend">
            {meta.featureImportance.slice(0, 10).map((f) => (
              <div key={f.feature}>
                <div style={{ display: 'flex', fontSize: 11.5, marginBottom: 3 }}>
                  <span>{f.label}</span>
                  <span className="mono muted" style={{ marginLeft: 'auto' }}>{f.value.toFixed(4)}</span>
                </div>
                <div className="bar-track">
                  <i style={{
                    width: (f.value / meta.featureImportance[0].value) * 100 + '%',
                    background: theme.accent
                  }} />
                </div>
              </div>
            ))}
          </div>
          <p className="note">
            Knowledge graph nodes {meta.kg.nodes.toLocaleString()} &middot; edges {meta.kg.edges.toLocaleString()} &middot;
            GNN link AUC {meta.gnn.final_link_auc}
          </p>
        </div>
      </section>
    </div>
  );
}
