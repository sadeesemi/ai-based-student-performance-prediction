/**
 * Service layer for the Module 03 dashboard.
 *
 * The React app never talks to the model directly. It calls this service, which
 * by default reads the artefacts the training pipeline published into
 * `public/data` (so the demo runs with no server at all).
 *
 * Point it at the Flask blueprint instead by creating frontend/.env with:
 *   REACT_APP_USE_BACKEND=true
 *   REACT_APP_API_BASE=http://localhost:5000/api/recommendation
 * The component layer does not change - only this file does.
 */
const USE_BACKEND = String(process.env.REACT_APP_USE_BACKEND) === 'true';
const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000/api/recommendation';
const STATIC_BASE = `${process.env.PUBLIC_URL || ''}/data`;
export const REPORT_BASE = `${process.env.PUBLIC_URL || ''}/reports`;

const cache = new Map();

async function getJSON(url) {
  if (cache.has(url)) return cache.get(url);
  const p = fetch(url).then((res) => {
    if (!res.ok) throw new Error(`Request failed (${res.status}) for ${url}`);
    return res.json();
  });
  cache.set(url, p);
  try {
    return await p;
  } catch (err) {
    cache.delete(url);
    throw err;
  }
}

export function getMeta() {
  return getJSON(USE_BACKEND ? `${API_BASE}/meta` : `${STATIC_BASE}/meta.json`);
}

export function getIndex() {
  return getJSON(USE_BACKEND ? `${API_BASE}/students` : `${STATIC_BASE}/students_index.json`);
}

export async function getStudent(id) {
  const sid = String(id || '').trim().toUpperCase();
  if (!sid) throw new Error('No student id supplied');
  if (USE_BACKEND) return getJSON(`${API_BASE}/student/${sid}`);
  const index = await getIndex();
  const row = index.find((s) => s.id === sid);
  if (!row) throw new Error(`Student ${sid} is not in the trained cohort`);
  const shard = await getJSON(`${STATIC_BASE}/students/shard_${String(row.shard).padStart(2, '0')}.json`);
  const student = shard[sid];
  if (!student) throw new Error(`Student ${sid} is missing from the published payload`);
  return student;
}

export async function searchStudents(query, limit = 8) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return [];
  const index = await getIndex();
  const starts = [];
  const contains = [];
  for (const s of index) {
    const id = s.id.toLowerCase();
    const name = s.name.toLowerCase();
    if (id.startsWith(q) || name.startsWith(q)) starts.push(s);
    else if (id.includes(q) || name.includes(q)) contains.push(s);
    if (starts.length >= limit) break;
  }
  return [...starts, ...contains].slice(0, limit);
}

export function reportUrl(file) {
  return `${REPORT_BASE}/${file}`;
}

export function openReport(file, params) {
  const qs = params ? `?${new URLSearchParams(params).toString()}` : '';
  window.open(`${reportUrl(file)}${qs}`, '_blank', 'noopener,width=1500,height=940');
}

export function openKnowledgeGraph(studentId) {
  openReport('knowledge_graph.html', studentId ? { student: studentId } : null);
}

export function openVisualReport() {
  openReport('visual_report.html');
}

const api = { getMeta, getIndex, getStudent, searchStudents, reportUrl, openReport, openKnowledgeGraph, openVisualReport };
export default api;
