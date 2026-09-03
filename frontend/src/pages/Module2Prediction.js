import React from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis
} from 'recharts';
import { ChartTip, ErrorBox, Figures, Loading, Meter, NoStudent, PanelHead } from '../components/Bits';
import FeatureImpact from '../components/FeatureImpact';
import RiskBadge from '../components/RiskBadge';
import { useStudent } from '../StudentContext';
import theme from '../theme';

const BANDS = ['Low Risk', 'Medium Risk', 'High Risk'];
const GRADE_ORDER = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F'];

export default function Module2Prediction() {
  const { meta, metaError, student, loading, error, selectStudent } = useStudent();
  if (metaError) return <ErrorBox message={'Could not load trained output: ' + metaError} />;
  if (!meta) return <Loading />;

  const m2 = meta.module2;
  const riskRows = BANDS.map((b) => ({ name: b, value: m2.riskDistribution[b] || 0 }));
  const gradeRows = GRADE_ORDER.filter((g) => m2.gradeDistribution[g])
    .map((g) => ({ name: g, value: m2.gradeDistribution[g] }));
  const bandRow = student ? m2.featureMeans.find((f) => f.band === student.riskLevel) : null;
  const passRate = ((m2.passFail.Pass || 0) / meta.nStudents) * 100;

  return (
    <div className="stack stack-36">
      {loading ? <Loading /> : null}
      <ErrorBox message={error} />

      {!student ? (
        <NoStudent
          title="Risk prediction is per learner"
          body="Search a student ID above to see the predicted risk band, how that learner compares with their band average and which drivers carry into the intervention ranker. Cohort prediction analytics stay visible below."
          onPick={selectStudent}
        />
      ) : (
        <>
          <section className="identity">
            <div className="identity-main">
              <div className="avatar">{student.name.split(' ').map((n) => n[0]).slice(0, 2).join('')}</div>
              <div>
                <div className="identity-name">{student.name}</div>
                <div className="identity-meta mono">{student.id} / {student.segment} / {student.moduleId}</div>
              </div>
            </div>
            <div className="identity-stats">
              <div className="identity-stat"><span className="v">{student.riskScore.toFixed(2)}</span><span className="l">Risk score</span></div>
              <div className="identity-stat"><span className="v">{student.totalMarks}</span><span className="l">Total marks</span></div>
              <div className="identity-stat"><span className="v">{student.grade}</span><span className="l">Grade</span></div>
              <div className="identity-stat">
                <span className="v" style={{ color: student.passFail === 'Pass' ? theme.low : theme.high }}>{student.passFail}</span>
                <span className="l">Outcome</span>
              </div>
            </div>
            <RiskBadge level={student.riskLevel} />
          </section>

          <section className="grid grid--wide-left">
            <div className="panel">
              <PanelHead title="Assessment and engagement record" note="Learner against the average of their predicted risk band" />
              <table className="tbl">
                <thead><tr><th>Metric</th><th className="num">Learner</th><th className="num">Band avg</th><th>Position</th></tr></thead>
                <tbody>
                  {[
                    ['Total marks', student.totalMarks, bandRow ? bandRow.totalMarks : '-', student.totalMarks],
                    ['Exam result', student.examResults, '-', student.examResults],
                    ['Quiz score', student.quizScores, bandRow ? bandRow.quiz : '-', student.quizScores],
                    ['CA marks (of 30)', student.caMarks, '-', (student.caMarks / 30) * 100],
                    ['Attendance %', student.attendancePct, bandRow ? bandRow.attendance : '-', student.attendancePct],
                    ['Engagement index', student.engagement, bandRow ? bandRow.engagement : '-', student.engagement],
                    ['Performance index', student.performance, bandRow ? bandRow.performance : '-', student.performance],
                    [student.assessmentType + ' score', student.assessmentScore + '/' + student.assessmentMax, '-', (student.assessmentScore / student.assessmentMax) * 100]
                  ].map((r) => (
                    <tr key={r[0]}>
                      <td>{r[0]}</td>
                      <td className="num">{r[1]}</td>
                      <td className="num muted">{r[2]}</td>
                      <td><Meter value={r[3]} width={110} color={r[3] > 60 ? theme.low : r[3] > 40 ? theme.medium : theme.high} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="panel">
              <PanelHead title="Risk drivers used downstream" note="Additive contributions carried into the intervention ranker" />
              <FeatureImpact data={student.features.slice(0, 7)} height={230} />
              <p className="note">
                The recommendation module consumes this risk state directly, so the intervention and the predicted
                score always move together.
              </p>
            </div>
          </section>
        </>
      )}

      <Figures
        items={[
          { label: 'Learners scored', value: meta.nStudents.toLocaleString() },
          { label: 'High risk', value: (m2.riskDistribution['High Risk'] || 0).toLocaleString() },
          { label: 'Medium risk', value: (m2.riskDistribution['Medium Risk'] || 0).toLocaleString() },
          { label: 'Low risk', value: (m2.riskDistribution['Low Risk'] || 0).toLocaleString() },
          { label: 'Pass rate', value: passRate.toFixed(1) + '%' }
        ]}
      />

      <section className="grid grid--3">
        <div className="panel">
          <PanelHead title="Risk band distribution" note="Module 02 output across the cohort" />
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={riskRows} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke="#f0ede6" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: theme.inkSoft }} />
              <YAxis tick={{ fontSize: 10, fill: theme.inkSoft }} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="value" name="learners" radius={[4, 4, 0, 0]} isAnimationActive={false} barSize={44}>
                {riskRows.map((r) => <Cell key={r.name} fill={theme.riskColor[r.name]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="panel">
          <PanelHead title="Risk by segment" note="Profiling segment against predicted band" />
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={m2.riskByStyle} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke="#f0ede6" vertical={false} />
              <XAxis dataKey="segment" tick={{ fontSize: 8.5, fill: theme.inkSoft }} interval={0} />
              <YAxis tick={{ fontSize: 10, fill: theme.inkSoft }} />
              <Tooltip content={<ChartTip />} />
              <Legend verticalAlign="bottom" iconType="circle" formatter={(v) => <span style={{ fontSize: 10, color: theme.inkSoft }}>{v}</span>} />
              {BANDS.map((b) => (
                <Bar key={b} dataKey={b} stackId="a" fill={theme.riskColor[b]} isAnimationActive={false} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="panel">
          <PanelHead title="Final grade spread" note="Recorded outcome distribution" />
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={gradeRows} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke="#f0ede6" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: theme.inkSoft }} />
              <YAxis tick={{ fontSize: 10, fill: theme.inkSoft }} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="value" name="learners" fill={theme.purple} radius={[4, 4, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel panel--flush">
        <div style={{ padding: '18px 20px 4px' }}>
          <PanelHead title="Band signatures" note="Average academic and behavioural profile per predicted risk band" />
        </div>
        <table className="tbl">
          <thead>
            <tr>
              <th>Risk band</th><th className="num">Learners</th><th className="num">Total marks</th>
              <th className="num">Quiz</th><th className="num">Attendance</th>
              <th className="num">Engagement</th><th className="num">Performance</th>
            </tr>
          </thead>
          <tbody>
            {m2.featureMeans.map((f) => (
              <tr key={f.band}>
                <td><RiskBadge level={f.band} size="sm" /></td>
                <td className="num">{f.learners}</td>
                <td className="num">{f.totalMarks}</td>
                <td className="num">{f.quiz}</td>
                <td className="num">{f.attendance}%</td>
                <td className="num">{f.engagement}</td>
                <td className="num">{f.performance}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
