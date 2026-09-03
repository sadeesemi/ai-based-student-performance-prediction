import React from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis
} from 'recharts';
import { ChartTip, ErrorBox, Figures, Loading, Meter, NoStudent, PanelHead } from '../components/Bits';
import RiskBadge, { SegmentBadge } from '../components/RiskBadge';
import { useStudent } from '../StudentContext';
import theme from '../theme';

export default function Module1Profiling() {
  const { meta, metaError, student, loading, error, selectStudent } = useStudent();
  if (metaError) return <ErrorBox message={'Could not load trained output: ' + metaError} />;
  if (!meta) return <Loading />;

  const m1 = meta.module1;
  const styleRows = Object.entries(m1.styleDistribution).map(([name, value]) => ({ name, value }));
  const gapRows = [...meta.lessons].sort((a, b) => a.gapLearners - b.gapLearners);
  const scatterBySegment = styleRows.map((s) => ({
    segment: s.name,
    points: m1.scatter.filter((p) => p.s === s.name)
  }));

  return (
    <div className="stack stack-36">
      {loading ? <Loading /> : null}
      <ErrorBox message={error} />

      {!student ? (
        <NoStudent
          title="Profiling output is per learner"
          body="Search a student ID above to inspect their behavioural profile, segment membership and the engagement indicators the profiling module produced. Cohort segmentation stays visible below."
          onPick={selectStudent}
        />
      ) : (
        <>
          <section className="identity">
            <div className="identity-main">
              <div className="avatar">{student.name.split(' ').map((n) => n[0]).slice(0, 2).join('')}</div>
              <div>
                <div className="identity-name">{student.name}</div>
                <div className="identity-meta mono">{student.id} / {student.program} / {student.course}</div>
              </div>
            </div>
            <div className="identity-stats">
              <div className="identity-stat"><span className="v">{student.engagement}</span><span className="l">Engagement index</span></div>
              <div className="identity-stat"><span className="v">{student.performance}</span><span className="l">Performance index</span></div>
              <div className="identity-stat"><span className="v">{student.attendancePct}%</span><span className="l">Attendance</span></div>
              <div className="identity-stat"><span className="v">{student.gpa.toFixed(2)}</span><span className="l">GPA</span></div>
            </div>
            <SegmentBadge segment={student.segment} />
            <RiskBadge level={student.riskLevel} />
          </section>

          <section className="grid grid--wide-left">
            <div className="panel">
              <PanelHead title="Behavioural indicators" note="Extracted LMS behaviour used to build the learner profile" />
              <table className="tbl">
                <thead>
                  <tr><th>Indicator</th><th className="num">Value</th><th>Cohort position</th></tr>
                </thead>
                <tbody>
                  {[
                    ['Login frequency', student.loginFrequency, (student.loginFrequency / 25) * 100],
                    ['Time spent on LMS', student.timeSpentLms, student.timeSpentLms],
                    ['Resources accessed', student.resourcesAccessed, (student.resourcesAccessed / 42) * 100],
                    ['Clickstream events', student.clickCount, (student.clickCount / 500) * 100],
                    ['Forum / chat activity', student.forumActivity, (student.forumActivity / 15) * 100],
                    ['Sessions attended', student.sessionsAttended + '/' + student.sessionsHeld, (student.sessionsAttended / student.sessionsHeld) * 100],
                    ['Engagement percentile', student.engagementPercentile + '%', student.engagementPercentile],
                    ['Performance percentile', student.performancePercentile + '%', student.performancePercentile]
                  ].map((rowData) => (
                    <tr key={rowData[0]}>
                      <td>{rowData[0]}</td>
                      <td className="num">{rowData[1]}</td>
                      <td>
                        <Meter
                          value={rowData[2]}
                          width={120}
                          color={rowData[2] > 60 ? theme.low : rowData[2] > 35 ? theme.medium : theme.high}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="panel">
              <PanelHead title="Segment and gaps" note="Profile propagated to the prediction and recommendation modules" />
              <div className="kv"><span className="muted">Segment</span><b>{student.segment}</b></div>
              <div className="kv"><span className="muted">Predicted risk</span><b>{student.riskLevel}</b></div>
              <div className="kv"><span className="muted">Previous results</span><b>{student.priorResults}</b></div>
              <div className="kv"><span className="muted">Submission behaviour</span><b>{student.submissionStatus}{student.delayDays > 0 ? ' (+' + student.delayDays + 'd)' : ''}</b></div>
              <div className="kv"><span className="muted">Intervention priority</span><b>{student.priority.toFixed(3)}</b></div>
              <p className="note" style={{ marginBottom: 8 }}>Non-engaged lessons</p>
              <div className="row">
                {student.gapLessons.length === 0
                  ? <span className="muted">None flagged - engagement recorded across every lesson.</span>
                  : student.gapLessons.map((g) => <span className="badge badge--plain" key={g}>{g}</span>)}
              </div>
            </div>
          </section>
        </>
      )}

      <Figures
        items={[
          { label: 'Learners profiled', value: meta.nStudents.toLocaleString() },
          { label: 'Segments', value: Object.keys(m1.styleDistribution).length },
          { label: 'Learners with gaps', value: meta.preprocessing.learners_with_gaps.toLocaleString() },
          { label: 'Fully engaged', value: meta.preprocessing.engaged_learners.toLocaleString() },
          { label: 'Lessons tracked', value: meta.lessons.length }
        ]}
      />

      <section className="grid grid--wide-right">
        <div className="panel">
          <PanelHead title="Segment distribution" note="Learner categories from the profiling module" />
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={styleRows} dataKey="value" nameKey="name" innerRadius={50} outerRadius={84}
                paddingAngle={2} stroke={theme.paper} strokeWidth={2} isAnimationActive={false}>
                {styleRows.map((s) => <Cell key={s.name} fill={theme.segmentColor[s.name] || theme.accent} />)}
              </Pie>
              <Tooltip content={<ChartTip />} />
              <Legend verticalAlign="bottom" iconType="circle"
                formatter={(v) => <span style={{ fontSize: 11, color: theme.inkSoft }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="panel">
          <PanelHead title="Engagement vs performance" note="Learner space coloured by segment (sampled for display)" />
          <ResponsiveContainer width="100%" height={250}>
            <ScatterChart margin={{ top: 6, right: 14, bottom: 4, left: -14 }}>
              <CartesianGrid stroke="#f0ede6" />
              <XAxis type="number" dataKey="x" name="engagement" tick={{ fontSize: 10, fill: theme.inkSoft }} />
              <YAxis type="number" dataKey="y" name="performance" tick={{ fontSize: 10, fill: theme.inkSoft }} />
              <ZAxis range={[16, 16]} />
              <Tooltip content={<ChartTip />} />
              <Legend verticalAlign="bottom" iconType="circle"
                formatter={(v) => <span style={{ fontSize: 10.5, color: theme.inkSoft }}>{v}</span>} />
              {scatterBySegment.map((s) => (
                <Scatter key={s.segment} name={s.segment} data={s.points}
                  fill={theme.segmentColor[s.segment] || theme.accent} isAnimationActive={false} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="grid grid--wide-left">
        <div className="panel panel--flush">
          <div style={{ padding: '18px 20px 4px' }}>
            <PanelHead title="Segment profile" note="Average behaviour per learner category" />
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>Segment</th><th className="num">Learners</th><th className="num">Engagement</th>
                <th className="num">Performance</th><th className="num">Attendance</th>
                <th className="num">Logins</th><th className="num">Gaps</th>
              </tr>
            </thead>
            <tbody>
              {m1.segmentProfile.map((s) => (
                <tr key={s.segment}>
                  <td><SegmentBadge segment={s.segment} /></td>
                  <td className="num">{s.learners}</td>
                  <td className="num">{s.engagement}</td>
                  <td className="num">{s.performance}</td>
                  <td className="num">{s.attendance}%</td>
                  <td className="num">{s.logins}</td>
                  <td className="num">{s.gaps}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <PanelHead title="Non-engaged lessons" note="Learners flagged per lesson" />
          <ResponsiveContainer width="100%" height={290}>
            <BarChart data={gapRows} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 4 }}>
              <XAxis type="number" tick={{ fontSize: 10, fill: theme.inkSoft }} />
              <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 9.5, fill: theme.ink }} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="gapLearners" name="learners" fill={theme.accent} radius={[0, 3, 3, 0]} isAnimationActive={false} barSize={12} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
