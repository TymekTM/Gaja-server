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
        li.innerHTML = `
          <div class="title">${t.name}</div>
          <div class="kv">
            <div>Plugin</div><div class="code">${t.plugin || '-'}</div>
            <div>Description</div><div>${t.description || ''}</div>
          </div>
        `;
        availableEl.appendChild(li);
      });
    } catch (e) {
      console.error(e);
    }
  }

  function addMsg(role, text) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.textContent = text;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

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

  async function sendQuery() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMsg('user', text);
    sendBtn.disabled = true;
    try {
      const r = await fetch('/api/v1/debug/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text })
      });
      const data = await r.json();
      if (!r.ok) {
        addMsg('assistant', `(Błąd) ${data?.error || 'Żądanie nieudane'}`);
        return;
      }
      if (data.model) modelEl.textContent = `Model: ${data.model}`;
      if (data.provider) providerEl.textContent = `Provider: ${data.provider}${data.fallbackUsed ? ' (fallback)' : ''}`;
      addMsg('assistant', data.response || '(brak odpowiedzi)');
      renderToolCalls(data.toolCalls || []);
      renderTrace(data.traceEvents || []);
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
  input.focus();
})();

