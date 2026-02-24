const form = document.getElementById('intake-form');
const timeline = document.getElementById('timeline');
const API_BASE = 'http://localhost:8000';

function addTimelineItem(text) {
  const li = document.createElement('li');
  li.textContent = text;
  timeline.appendChild(li);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  timeline.innerHTML = '';

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());
  if (!payload.github_repo_url) delete payload.github_repo_url;
  if (!payload.seed_question) delete payload.seed_question;

  const projectRes = await fetch(`${API_BASE}/api/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const projectJson = await projectRes.json();
  const projectId = projectJson.project.id;
  addTimelineItem(`Project created: ${projectId}`);

  const runRes = await fetch(`${API_BASE}/api/projects/${projectId}/runs`, { method: 'POST' });
  const runJson = await runRes.json();
  const runId = runJson.run.id;
  addTimelineItem(`Run started: ${runId}`);

  const timelineRes = await fetch(`${API_BASE}/api/runs/${runId}/timeline`);
  const timelineJson = await timelineRes.json();
  timelineJson.events.forEach((entry) => {
    addTimelineItem(`${entry.event_type}: ${entry.message} (c=${entry.confidence})`);
  });
});
