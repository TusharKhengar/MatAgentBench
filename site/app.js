/* MatAgentBench viewer.
 *
 * Pure renderer over committed JSON. There is no backend and no API key anywhere in
 * this file by design -- everything on GitHub Pages is public, so all model calls
 * happen in GitHub Actions with keys in repository secrets, and the browser only ever
 * reads results that were computed there.
 */

const RESULTS = "results";
const SILENT = new Set([
  "silent_unit", "silent_basis", "silent_cell", "silent_state", "silent_sign",
]);

const state = { index: null, taskset: null, leaderboard: null, selected: null, tasks: new Map() };

const $ = (sel) => document.querySelector(sel);
const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function loadJSON(path) {
  for (const prefix of ["results", "../results", "/results"]) {
    try {
      const response = await fetch(`${prefix}/${path}`, { cache: "no-cache" });
      if (response.ok) return await response.json();
    } catch {
      // try next prefix
    }
  }
  throw new Error(`${path}: not found`);
}

function badgeClass(failureClass, passed) {
  if (failureClass === "unearned_pass") return "unearned";
  if (passed) return "pass";
  return SILENT.has(failureClass) ? "silent" : "fail";
}

/* ------------------------------ headline ------------------------------- */

function renderHeadline() {
  const lb = state.leaderboard;
  if (!lb) return;
  const head = lb.headline || {};
  const dominant = head.dominant_failure;

  const stats = [
    { value: pct(head.silent_failure_share_of_failures), label: "of all failures are silent", headline: true },
    { value: String(lb.n_tasks ?? 0), label: "calibrated tasks" },
    { value: String(head.n_models ?? 0), label: "models evaluated" },
    { value: String(head.unearned_passes ?? 0), label: "unearned passes caught" },
  ];

  $("#stats").innerHTML = stats.map((s) => `
    <div class="stat${s.headline ? " is-headline" : ""}">
      <div class="value">${esc(s.value)}</div>
      <div class="label">${esc(s.label)}</div>
    </div>`).join("");

  $("#callout").innerHTML = dominant
    ? `The single most common failure is <strong>${esc(dominant.failure_class)}</strong>,
       accounting for <strong>${pct(dominant.share)}</strong> of all failures. Silent
       failures raise no exception and pass every syntactic check &mdash; a plain
       correctness test would report them only as &ldquo;wrong&rdquo;.`
    : "No failures recorded yet.";

  $("#generated").textContent =
    `${lb.benchmark} · task set ${lb.taskset} · generated ${new Date(lb.generated_at).toLocaleString()}`;
}

/* ----------------------------- leaderboard ----------------------------- */

function renderLeaderboard() {
  const rows = (state.leaderboard?.results || []).map((r) => `
    <tr>
      <td>${esc(r.model.backend)}/${esc(r.model.model)}</td>
      <td>${r.model.open_weights ? "open" : "closed"}</td>
      <td class="num">${pct(r.pass_rate)}</td>
      <td class="num">${pct(r.earned_pass_rate)}</td>
      <td class="num">${pct(r.silent_failure_rate)}</td>
      <td class="num">${r.mean_steps ? r.mean_steps.toFixed(1) : "—"}</td>
    </tr>`).join("");
  $("#leaderboard-table tbody").innerHTML =
    rows || `<tr><td colspan="6">No results yet.</td></tr>`;
}

/* ------------------------------ taxonomy ------------------------------- */

function renderTaxonomy() {
  const totals = {};
  for (const result of state.leaderboard?.results || []) {
    for (const [cls, n] of Object.entries(result.failure_counts || {})) {
      if (cls === "success") continue;
      totals[cls] = (totals[cls] || 0) + n;
    }
  }
  const entries = Object.entries(totals).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, n]) => n));

  $("#taxonomy-chart").innerHTML = entries.length
    ? entries.map(([cls, n]) => `
        <div class="bar-row">
          <span class="name">${esc(cls)}</span>
          <span class="bar-track">
            <span class="bar-fill ${SILENT.has(cls) ? "silent" : ""}"
                  style="width:${(n / max) * 100}%"></span>
          </span>
          <span class="num">${n}</span>
        </div>`).join("")
    : `<p class="lede">No failures recorded yet.</p>`;
}

/* ----------------------------- attribution ----------------------------- */

function renderAttribution() {
  const attempted = {}, recovered = {};
  for (const result of state.leaderboard?.results || []) {
    const attribution = result.attribution;
    if (!attribution) continue;
    for (const [kind, n] of Object.entries(attribution.attempted_by_kind || {})) {
      attempted[kind] = (attempted[kind] || 0) + n;
    }
    for (const [kind, n] of Object.entries(attribution.recovered_by_kind || {})) {
      recovered[kind] = (recovered[kind] || 0) + n;
    }
  }
  const kinds = Object.keys(attempted).sort();

  $("#attribution-chart").innerHTML = kinds.length
    ? kinds.map((kind) => {
        const rate = (recovered[kind] || 0) / attempted[kind];
        return `
          <div class="bar-row">
            <span class="name">${esc(kind)}</span>
            <span class="bar-track">
              <span class="bar-fill pass" style="width:${rate * 100}%"></span>
            </span>
            <span class="num">${pct(rate)}</span>
          </div>
          <div class="bar-row">
            <span class="name"></span>
            <span class="num" style="grid-column: 2 / span 2">
              ${recovered[kind] || 0} of ${attempted[kind]} failures recovered
            </span>
          </div>`;
      }).join("")
    : `<p class="lede">No attribution runs yet. Run <code>mab attribute</code>.</p>`;
}

/* ------------------------------- viewer -------------------------------- */

function filteredEntries() {
  const model = $("#filter-model").value;
  const failure = $("#filter-failure").value;
  const tier = $("#filter-tier").value;
  return (state.index?.entries || []).filter((e) =>
    (!model || e.model === model) &&
    (!failure || e.failure_class === failure) &&
    (!tier || String(e.tier) === tier));
}

function renderRunList() {
  const entries = filteredEntries();
  $("#run-count").textContent = `${entries.length} run${entries.length === 1 ? "" : "s"}`;
  $("#run-list").innerHTML = entries.map((e, i) => `
    <li data-i="${i}" role="option" aria-selected="${state.selected === e.path}" tabindex="0">
      <span class="task">${esc(e.task_id)}</span>
      <span class="meta">
        <span class="badge ${badgeClass(e.failure_class, e.passed)}">${esc(e.failure_class)}</span>
        · tier ${e.tier} · ${e.n_steps} steps${e.has_counterfactual ? " · ↻" : ""}
      </span>
    </li>`).join("") || `<li class="empty">No runs match these filters.</li>`;

  $("#run-list").querySelectorAll("li[data-i]").forEach((li) => {
    const entry = entries[Number(li.dataset.i)];
    const open = () => selectRun(entry);
    li.addEventListener("click", open);
    li.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); open(); }
    });
  });
}

async function selectRun(entry) {
  state.selected = entry.path;
  renderRunList();
  $("#run-detail").innerHTML = `<p class="empty">Loading ${esc(entry.task_id)}…</p>`;
  try {
    const traj = await loadJSON(entry.path);
    renderRunDetail(traj, entry);
  } catch (err) {
    $("#run-detail").innerHTML =
      `<p class="empty">Could not load this trajectory (${esc(err.message)}).</p>`;
  }
}

function renderVerdict(verdict) {
  if (!verdict) return "";
  const cls = badgeClass(verdict.failure_class, verdict.passed);
  const rec = verdict.reconciliation;
  return `
    <div class="verdict ${cls === "unearned" ? "fail" : cls}">
      <h4>
        <span class="badge ${cls}">${esc(verdict.failure_class)}</span>
        ${verdict.passed ? "" : " — the agent's answer was rejected"}
      </h4>
      ${verdict.detail ? `<p>${esc(verdict.detail)}</p>` : ""}
      ${rec && rec.matched
        ? `<div class="factor">reconciled by: ${esc(rec.label)}</div>`
        : ""}
    </div>`;
}

function renderStep(step) {
  const call = step.tool_call;
  const result = step.tool_result;
  const failed = result && !result.ok;
  return `
    <details class="step">
      <summary>
        <span class="idx">${step.index}</span>
        <span class="tool">${call ? esc(call.name) : "— final answer —"}</span>
        ${failed ? `<span class="err">${esc(result.error || "error")}</span>` : ""}
        ${step.context_tokens ? `<span class="err" style="margin-left:auto;color:var(--muted)">~${step.context_tokens} ctx</span>` : ""}
      </summary>
      <div class="step-body">
        ${step.thought ? `<h5>Thought</h5><pre>${esc(step.thought.trim())}</pre>` : ""}
        ${call ? `<h5>Call</h5><pre>${esc(call.name)}(${esc(JSON.stringify(call.args))})</pre>` : ""}
        ${result ? `<h5>Observation</h5><pre>${esc(
            (result.ok ? result.content : result.error) || "").slice(0, 4000)}</pre>` : ""}
      </div>
    </details>`;
}

function renderRunDetail(traj, entry) {
  const task = state.tasks.get(traj.task_id);
  const verdict = traj.verdict;
  const reported = verdict?.reported;
  const truth = verdict?.truth;

  $("#run-detail").innerHTML = `
    <h3 style="margin:0 0 4px;font-size:16px;font-family:var(--mono)">${esc(traj.task_id)}</h3>
    <p class="sub" style="margin:0;color:var(--muted);font-size:13px">
      ${esc(entry.model)} · tier ${entry.tier} · ${traj.steps.length} steps ·
      ${traj.total_tokens} tokens
    </p>

    ${task ? `<div class="prompt">${esc(task.prompt)}</div>` : ""}

    ${renderVerdict(verdict)}

    <dl class="kv">
      ${reported ? `<dt>Agent reported</dt><dd>${esc(reported.value)} ${esc(reported.unit)}
        [${esc(reported.basis)}, ${esc(reported.cell)}, ${esc(reported.state)}]</dd>` : ""}
      ${truth ? `<dt>Ground truth</dt><dd>${esc(truth.value)} ${esc(truth.unit)}
        [${esc(truth.basis)}, ${esc(truth.cell)}, ${esc(truth.state)}]</dd>` : ""}
      ${verdict?.relative_error != null
        ? `<dt>Relative error</dt><dd>${pct(verdict.relative_error)}</dd>` : ""}
      ${task ? `<dt>Required stages</dt><dd>${esc((task.required_stages || []).join(" → "))}</dd>` : ""}
    </dl>

    ${verdict?.stage_checks && Object.keys(verdict.stage_checks).length ? `
      <dl class="kv">
        <dt>Stage checks</dt>
        <dd>${Object.entries(verdict.stage_checks)
          .map(([k, v]) => `${v ? "✓" : "✗"} ${esc(k)}`).join("<br>")}</dd>
      </dl>` : ""}

    <div class="steps">
      <h4 style="font-size:14px;margin:0 0 10px">Trajectory</h4>
      ${traj.steps.map(renderStep).join("")}
    </div>

    <div class="cf" id="cf-slot">
      ${entry.has_counterfactual
        ? `<h4>Counterfactual repairs</h4><p class="empty">Loading…</p>`
        : ""}
    </div>`;

  if (entry.has_counterfactual) loadCounterfactuals(traj, entry);
}

async function loadCounterfactuals(traj, entry) {
  const modelSlug = entry.path.split("/")[1];
  const kinds = ["convention_correct", "plan_repair", "context_restore", "tool_repair"];
  const found = [];

  for (const kind of kinds) {
    for (let k = 0; k <= 30; k++) {
      try {
        const cf = await loadJSON(
          `counterfactuals/${modelSlug}/${entry.task_id}__${kind}__k${k}.json`);
        found.push({ kind, k, cf });
        break;
      } catch { /* that (kind, k) pair was not run */ }
    }
  }

  const slot = $("#cf-slot");
  if (!found.length) {
    slot.innerHTML = `<h4>Counterfactual repairs</h4>
      <p class="empty">No counterfactual runs found for this trajectory.</p>`;
    return;
  }

  slot.innerHTML = `<h4>Counterfactual repairs</h4>` + found.map(({ kind, k, cf }) => {
    const recovered = cf.verdict?.passed && cf.verdict.failure_class !== "unearned_pass";
    return `
      <div class="cf-item">
        <span class="kind">${esc(kind)}</span>
        <span>at step ${k}</span>
        <span class="badge ${recovered ? "pass" : "fail"}">
          ${recovered ? "recovered" : "still failed"}
        </span>
        <span style="color:var(--muted)">
          ${esc(cf.verdict?.failure_class || "—")}
        </span>
        <span class="rationale">
          ${recovered
            ? "Repairing only this one thing turned the failure into a pass — evidence that it was the cause."
            : "Repairing this did not recover the run, so it was not the binding constraint."}
        </span>
      </div>`;
  }).join("");
}

/* -------------------------------- boot --------------------------------- */

function populateFilters() {
  const models = [...new Set((state.index?.entries || []).map((e) => e.model))].sort();
  const failures = [...new Set((state.index?.entries || []).map((e) => e.failure_class))].sort();
  const fill = (sel, values) => {
    for (const v of values) {
      const option = document.createElement("option");
      option.value = v; option.textContent = v;
      $(sel).appendChild(option);
    }
  };
  fill("#filter-model", models);
  fill("#filter-failure", failures);
  for (const sel of ["#filter-model", "#filter-failure", "#filter-tier"]) {
    $(sel).addEventListener("change", renderRunList);
  }
}

function setRepoLink() {
  const [owner, repo] = location.hostname.endsWith("github.io")
    ? [location.hostname.split(".")[0], location.pathname.split("/").filter(Boolean)[0]]
    : [null, null];
  $("#repo-link").href = owner && repo
    ? `https://github.com/${owner}/${repo}`
    : "https://github.com";
}

async function main() {
  setRepoLink();
  try {
    const [leaderboard, index, taskset] = await Promise.all([
      loadJSON("leaderboard.json"),
      loadJSON("trajectories/index.json"),
      loadJSON("taskset.json").catch(() => null),
    ]);
    state.leaderboard = leaderboard;
    state.index = index;
    state.taskset = taskset;
    for (const task of taskset?.tasks || []) state.tasks.set(task.task_id, task);

    renderHeadline();
    renderLeaderboard();
    renderTaxonomy();
    renderAttribution();
    populateFilters();
    renderRunList();

    const first = filteredEntries()[0];
    if (first) selectRun(first);
  } catch (err) {
    $("#generated").textContent =
      `No results published yet (${err.message}). Run the pipeline, then \`mab report\`.`;
  }
}

main();
