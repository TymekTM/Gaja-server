(function () {
  const q = (sel) => document.querySelector(sel);
  const byId = (id) => document.getElementById(id);

  const chatLog = byId('chat-log');
  const input = byId('query');
  const sendBtn = byId('send');
  const toolCallsEl = byId('tool-calls');
  const traceEl = byId('trace-events');
  const availableEl = byId('available-tools');
  const modelEl = byId('model');
  const providerEl = byId('provider');
  const usageEl = byId('usage');
  const latencyEl = byId('latency');
  const sysPromptEl = byId('system-prompt');
  const msgsEl = byId('messages-json');
  const providerSelect = byId('providerSelect');
  const modelInput = byId('modelInput');
  const noFallbackCb = byId('noFallback');
  const lmstudioUrlInput = byId('lmstudioUrl');
  const saveLmstudioBtn = byId('saveLmstudioUrl');
  let providerDefaults = {};

  // Tabs
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.pane').forEach((p) => p.classList.remove('active'));
      tab.classList.add('active');
      byId('tab-' + tab.dataset.tab).classList.add('active');
    });
  });

  async function fetchAvailableTools() {
    try {
      const r = await fetch('/api/v1/debug/tools');
      if (!r.ok) return;
      const data = await r.json();
      availableEl.innerHTML = '';
      (data.tools || []).forEach((t) => {
        const li = document.createElement('li');
        li.className = 'item';
        const stale = t.stale ? ' <span style="color:#f59e0b">(stale)</span>' : '';
        const params = t.parameters ? JSON.stringify(t.parameters, null, 2) : '{}';
        li.innerHTML = `
          <div class="title">${t.name}${stale}</div>
          <div class="kv">
            <div>Plugin</div><div class="code">${t.plugin || '-'}</div>
            <div>Description</div><div>${t.description || ''}</div>
            <div>Parameters</div><div class="code">${params}</div>
            <div>Tested OK</div>
            <div>
              <label style="display:flex;align-items:center;gap:8px;">
                <input type="checkbox" class="verify" ${t.tested ? 'checked' : ''} data-name="${t.name}">
                <span>${t.tested ? 'yes' : 'no'}</span>
              </label>
              <div style="font-size:11px;opacity:0.7;">last: ${t.lastVerifiedVersion || '-'} @ ${t.lastVerifiedAt || '-'}</div>
            </div>
          </div>
        `;
        // Attach checkbox handler
        const cb = li.querySelector('input.verify');
        const label = li.querySelector('label span');
        cb.addEventListener('change', async () => {
          try {
            cb.disabled = true;
            const resp = await fetch('/api/v1/debug/tools/verify', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ name: t.name, tested: cb.checked })
            });
            const j = await resp.json();
            label.textContent = cb.checked ? 'yes' : 'no';
          } catch (e) {
            console.error(e);
            cb.checked = !cb.checked;
          } finally {
            cb.disabled = false;
          }
        });
        availableEl.appendChild(li);
      });
    } catch (e) {
      console.error(e);
    }
  }

  async function fetchProviders() {
    try {
      const r = await fetch('/api/v1/debug/providers');
      let data = null;
      if (r.ok) {
        data = await r.json();
      }
      let providers = (data && data.providers) || [];
      providerDefaults = (data && data.defaults) || {};
      const baseUrls = (data && data.baseUrls) || {};
      // Hard fallback if backend unavailable or returned empty list
      if (!Array.isArray(providers) || providers.length === 0) {
        providers = ['openai', 'openrouter'];
        providerDefaults = Object.assign({
          openai: 'gpt-5-nano',
          openrouter: 'openai/gpt-oss-20b:free',
        }, providerDefaults || {});
      }
      // Populate select
      providerSelect.innerHTML = '';
      providers.forEach((p) => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p;
        providerSelect.appendChild(opt);
      });
      // Set current selection and model
      const current = (data && data.current) || {};
      if (current.provider && providers.includes(current.provider)) {
        providerSelect.value = current.provider;
      }
      const initialModel = providerDefaults[providerSelect.value] || current.model || '';
      modelInput.value = initialModel;
      // LM Studio base URL
      if (lmstudioUrlInput && baseUrls && baseUrls.lmstudio) {
        lmstudioUrlInput.value = baseUrls.lmstudio || '';
      }
      // Reflect in meta labels initially
      if (current.model) modelEl.textContent = `Model: ${current.model}`;
      if (current.provider) providerEl.textContent = `Provider: ${current.provider}`;
    } catch (e) {
      console.error(e);
      // Last-resort hard fallback
      providerSelect.innerHTML = '';
      ['openai', 'openrouter'].forEach((p) => {
        const opt = document.createElement('option');
        opt.value = p; opt.textContent = p; providerSelect.appendChild(opt);
      });
      providerDefaults = { openai: 'gpt-5-nano', openrouter: 'openai/gpt-oss-20b:free' };
      providerSelect.value = 'openai';
      modelInput.value = providerDefaults['openai'];
    }
  }

  function addMsg(role, text) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.textContent = text;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  providerSelect?.addEventListener('change', () => {
    const p = providerSelect.value;
    const suggested = providerDefaults[p];
    if (suggested) modelInput.value = suggested;
  });

  saveLmstudioBtn?.addEventListener('click', async () => {
    try {
      saveLmstudioBtn.disabled = true;
      const baseUrl = (lmstudioUrlInput?.value || '').trim();
      if (!baseUrl) return;
      const r = await fetch('/api/v1/debug/providers/lmstudio_url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseUrl })
      });
      const j = await r.json();
      if (!r.ok || !j.success) {
        alert('Nie udało się zapisać URL LM Studio: ' + (j.error || r.status));
      }
    } catch (e) {
      console.error(e);
      alert('Błąd zapisu URL LM Studio');
    } finally {
      saveLmstudioBtn.disabled = false;
    }
  });

  function renderToolCalls(calls) {
    toolCallsEl.innerHTML = '';
    (calls || []).forEach((c, idx) => {
      const li = document.createElement('li');
      li.className = 'item';
      const args = JSON.stringify(c.arguments ?? {}, null, 2);
      const result = typeof c.result === 'string' ? c.result : JSON.stringify(c.result ?? null, null, 2);
      li.innerHTML = `
        <div class="title">#${idx + 1} ${c.name || '(unknown function)'} </div>
        <div class="kv">
          <div>Arguments</div>
          <div class="code">${args}</div>
          <div>Result</div>
          <div class="code">${result}</div>
        </div>
      `;
      toolCallsEl.appendChild(li);
    });
  }

  function renderTrace(events) {
    traceEl.innerHTML = '';
    (events || []).forEach((ev) => {
      const li = document.createElement('li');
      li.className = 'item';
      li.innerHTML = `
        <div class="title">${ev.event}</div>
        <div class="kv">
          <div>t_rel_ms</div><div>${Math.round(ev.t_rel_ms)} ms</div>
          <div>epoch</div><div>${new Date(ev.t_epoch * 1000).toLocaleTimeString()}</div>
          <div>extra</div><div class="code">${ev.extra ? JSON.stringify(ev.extra, null, 2) : '-'}</div>
        </div>
      `;
      traceEl.appendChild(li);
    });
  }

  const sessionHistory = [];

  async function sendQuery() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMsg('user', text);
    sessionHistory.push({ role: 'user', content: text });
    sendBtn.disabled = true;
    try {
      const r = await fetch('/api/v1/debug/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          history: sessionHistory,
          providerOverride: providerSelect?.value || null,
          forceModel: (modelInput?.value || '').trim() || null,
          noFallback: !!(noFallbackCb && noFallbackCb.checked),
        })
      });
      const data = await r.json();
      if (!r.ok) {
        addMsg('assistant', `(Błąd) ${data?.error || 'Żądanie nieudane'}`);
        return;
      }
      if (data.model) modelEl.textContent = `Model: ${data.model}`;
      if (data.provider) providerEl.textContent = `Provider: ${data.provider}${data.fallbackUsed ? ' (fallback)' : ''}`;
      if (data.usage) {
        const p = data.usage.prompt_tokens ?? '-';
        const c = data.usage.completion_tokens ?? '-';
        const t = data.usage.total_tokens ?? '-';
        usageEl.textContent = `Tokens: p=${p} c=${c} t=${t}`;
      } else {
        usageEl.textContent = '';
      }
      if (typeof data.elapsedMs === 'number') {
        latencyEl.textContent = `Latency: ${data.elapsedMs} ms`;
      } else {
        latencyEl.textContent = '';
      }
      const reply = data.response || '(brak odpowiedzi)';
      addMsg('assistant', reply);
      sessionHistory.push({ role: 'assistant', content: reply });
      renderToolCalls(data.toolCalls || []);
      renderTrace(data.traceEvents || []);
      // Payload
      if (data.systemPrompt) sysPromptEl.textContent = data.systemPrompt;
      if (data.messages) msgsEl.textContent = JSON.stringify(data.messages, null, 2);
    } catch (e) {
      console.error(e);
      addMsg('assistant', '(Błąd) ' + String(e));
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendQuery(); });
  sendBtn.addEventListener('click', sendQuery);

  fetchAvailableTools();
  fetchProviders();
  input.focus();
})();
