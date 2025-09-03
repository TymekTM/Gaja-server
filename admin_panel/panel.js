/* GAJA Admin Panel JS */
const API_BASE = '/api/v1';
let authToken = null;
const navItems = [
  {id:'dashboard', label:'Dashboard'},
  {id:'users', label:'Użytkownicy'},
  {id:'devices', label:'Urządzenia'},
  {id:'config', label:'Konfiguracja'},
  {id:'logs', label:'Logi'}
];

function el(tag, cls='', html=''){ const e=document.createElement(tag); if(cls) e.className=cls; if(html) e.innerHTML=html; return e; }

function setNav(active){
  const nav = document.getElementById('nav');
  nav.innerHTML='';
  navItems.forEach(item=>{
    const b = el('button', `px-3 py-1 rounded text-xs font-medium ${active===item.id?'bg-emerald-600 text-white':'bg-slate-800 hover:bg-slate-700'}` , item.label);
    b.onclick=()=>{ loadView(item.id); };
    nav.appendChild(b);
  });
}

async function api(path, options={}){
  const headers = options.headers || {};
  if(authToken) headers['Authorization'] = `Bearer ${authToken}`;
  headers['Content-Type'] = 'application/json';
  const res = await fetch(`${API_BASE}${path}`, {...options, headers});
  if(!res.ok){ throw new Error(`${res.status} ${res.statusText}`); }
  return res.json();
}

function formatDuration(seconds){
  const h = Math.floor(seconds/3600), m = Math.floor((seconds%3600)/60);
  return `${h}h ${m}m`;
}

async function loadDashboard(){
  const container = document.getElementById('viewContainer');
  container.innerHTML = '<div class="card">Ładowanie...</div>';
  try {
    const stats = await api('/admin/stats');
    const metrics = await api('/metrics');
    container.innerHTML = `
      <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <div class="card"><h3 class="font-semibold mb-2">Użytkownicy</h3><p class="text-3xl font-bold">${stats.active_users}</p><p class="text-xs text-slate-400">Aktywni</p></div>
        <div class="card"><h3 class="font-semibold mb-2">Interakcje</h3><p class="text-3xl font-bold">${stats.total_interactions}</p><p class="text-xs text-slate-400">Łącznie</p></div>
        <div class="card"><h3 class="font-semibold mb-2">Plugins</h3><p class="text-3xl font-bold">${stats.total_plugins}</p><p class="text-xs text-slate-400">Wszystkie</p></div>
        <div class="card"><h3 class="font-semibold mb-2">CPU</h3><p class="text-3xl font-bold">${metrics.cpu}%</p></div>
        <div class="card"><h3 class="font-semibold mb-2">RAM</h3><p class="text-3xl font-bold">${(metrics.ram.used/1024/1024/1024).toFixed(2)}<span class='text-base font-normal'>GB</span></p></div>
        <div class="card"><h3 class="font-semibold mb-2">Uptime</h3><p class="text-3xl font-bold">${formatDuration(metrics.uptime)}</p></div>
      </div>
      <div class="card mt-6">
        <h3 class="font-semibold mb-3">Aktywność</h3>
        <ul class="text-sm space-y-1">${stats.recent_activity.map(a=>`<li class='flex items-start gap-2'><span class='badge'>${a.type}</span><span>${a.action}</span></li>`).join('')}</ul>
      </div>`;
  } catch(e){
    container.innerHTML = `<div class='card text-red-400'>Błąd: ${e.message}</div>`;
  }
}

async function loadUsers(){
  const c = document.getElementById('viewContainer');
  c.innerHTML = '<div class="card">Ładowanie użytkowników...</div>';
  try {
    const data = await api('/admin/users');
    c.innerHTML = `<div class='card overflow-auto scrollbar-thin'>
      <h3 class='font-semibold mb-3'>Użytkownicy (${data.users.length})</h3>
      <table class='table-grid text-xs md:text-sm min-w-[600px]'>
        <thead><tr><th>ID</th><th>Nazwa</th><th>Plugins</th><th>Źródło</th><th>WS</th><th></th></tr></thead>
        <tbody>${data.users.map(u=>{
          const uname = (u.username && u.username.trim()) ? u.username : '(brak)';
          const plugins = (u.enabled_plugins && u.enabled_plugins.length)? u.enabled_plugins.join(', ') : '<span class="text-slate-500">—</span>';
          const active = data.active_connections.includes(''+u.id);
          return `<tr><td>${u.id}</td><td>${uname}</td><td>${plugins}</td><td>${u.source||''}</td><td>${active?'<span class="status-online" title="WebSocket active">●</span>':''}</td><td><button data-user='${u.id}' class='disconnect text-xs bg-red-600/20 hover:bg-red-600/30 px-2 py-0.5 rounded'>Disconnect</button></td></tr>`;
        }).join('')}</tbody>
      </table>
      ${!data.users.length?'<p class="text-xs text-slate-400 mt-2">Brak użytkowników.</p>':''}
    </div>`;
    c.querySelectorAll('button.disconnect').forEach(b=>{
      b.onclick=async()=>{
        const id=b.dataset.user; if(!confirm('Disconnect user '+id+'?')) return;
        try{ await api(`/admin/users/${id}/disconnect`, {method:'POST'}); loadUsers(); }catch(e){ alert(e.message);} };
    });
  } catch(e){ c.innerHTML = `<div class='card text-red-400'>Błąd: ${e.message}</div>`; }
}

async function loadDevices(){
  const c = document.getElementById('viewContainer');
  c.innerHTML = '<div class="card">Ładowanie urządzeń...</div>';
  try {
    const data = await api('/admin/devices');
    c.innerHTML = `<div class='space-y-4'>
      <div class='card flex items-center gap-2'>
        <form id='newDeviceForm' class='flex flex-wrap gap-2 items-end'>
          <label class='text-xs'>Nazwa<br><input name='name' required class='bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm'></label>
          <label class='text-xs'>Typ<br><select name='type' class='bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm'><option>headless</option><option>web</option><option>mobile</option></select></label>
          <button class='bg-emerald-600 hover:bg-emerald-500 text-sm px-3 py-1 rounded'>Dodaj</button>
        </form>
      </div>
      <div class='card overflow-auto scrollbar-thin'>
        <h3 class='font-semibold mb-3'>Urządzenia (${data.devices.length})</h3>
        <table class='table-grid'>
          <thead><tr><th>ID</th><th>Nazwa</th><th>Typ</th><th>Status</th><th>Last Seen</th><th>API Key</th><th></th></tr></thead>
          <tbody>${data.devices.map(d=>`<tr><td>${d.id}</td><td>${d.name}</td><td>${d.type}</td><td><span class='${'status-'+d.status}'>${d.status}</span></td><td>${d.last_seen || ''}</td><td>${d.api_key_masked||''}</td><td><button data-id='${d.id}' class='regen text-xs bg-indigo-600/20 hover:bg-indigo-600/30 px-2 py-0.5 rounded'>Regenerate</button> <button data-id='${d.id}' class='del text-xs bg-red-600/20 hover:bg-red-600/30 px-2 py-0.5 rounded'>X</button></td></tr>`).join('')}</tbody>
        </table>
      </div>
    </div>`;
    const form = document.getElementById('newDeviceForm');
    form.onsubmit=async(e)=>{
      e.preventDefault();
      const fd = new FormData(form);
      const payload = {name: fd.get('name'), type: fd.get('type')};
      try{ await api('/admin/devices',{method:'POST', body: JSON.stringify(payload)}); loadDevices(); form.reset(); }catch(err){ alert(err.message); }
    };
    c.querySelectorAll('button.regen').forEach(b=> b.onclick=async()=>{ try{ const id=b.dataset.id; const r= await api(`/admin/devices/${id}/regenerate-key`, {method:'POST'}); alert('Nowy klucz: '+r.api_key); loadDevices(); }catch(e){alert(e.message);} });
    c.querySelectorAll('button.del').forEach(b=> b.onclick=async()=>{ if(!confirm('Usunąć urządzenie?'))return; try{ const id=b.dataset.id; await api(`/admin/devices/${id}`, {method:'DELETE'}); loadDevices(); }catch(e){ alert(e.message);} });
  } catch(e){ c.innerHTML = `<div class='card text-red-400'>Błąd: ${e.message}</div>`; }
}

async function loadConfig(){
  const c = document.getElementById('viewContainer');
  c.innerHTML = '<div class="card">Ładowanie konfiguracji...</div>';
  try {
    const data = await api('/admin/server-config');
    const cfg = data.config || {};
    const ai = cfg.ai || {};
    const tts = cfg.tts || {};
    const plugins = cfg.plugins || {};
    c.innerHTML = `<div class='space-y-4'>
      <div class='card'>
        <h3 class='font-semibold mb-2 flex items-center justify-between'>Konfiguracja (podgląd)
          <button id='refreshCfg' class='text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600'>Odśwież</button>
        </h3>
        <pre class='text-xs whitespace-pre-wrap max-h-72 overflow-auto scrollbar-thin bg-slate-800/60 p-3 rounded'>${escapeHtml(JSON.stringify(cfg,null,2))}</pre>
      </div>
      <div class='grid gap-4 md:grid-cols-2'>
        <div class='card space-y-2'>
          <h3 class='font-semibold'>AI</h3>
          <label class='text-xs block'>Domyślny model
            <input id='ai_default_model' value='${ai.default_model||''}' class='w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm'/>
          </label>
          <label class='text-xs block'>Temperatura
            <input id='ai_temperature' type='number' step='0.01' value='${ai.temperature??''}' class='w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm'/>
          </label>
          <button id='saveAI' class='bg-emerald-600 hover:bg-emerald-500 text-sm px-3 py-1 rounded'>Zapisz AI</button>
        </div>
        <div class='card space-y-2'>
          <h3 class='font-semibold'>TTS</h3>
          <label class='text-xs block'>Silnik
            <input id='tts_engine' value='${tts.engine||''}' class='w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm'/>
          </label>
            <label class='text-xs block'>Głos
            <input id='tts_voice' value='${tts.voice||''}' class='w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm'/>
          </label>
          <button id='saveTTS' class='bg-emerald-600 hover:bg-emerald-500 text-sm px-3 py-1 rounded'>Zapisz TTS</button>
        </div>
        <div class='card space-y-2 md:col-span-2'>
          <h3 class='font-semibold'>Plugins</h3>
          <label class='text-xs block'>Lista domyślnie włączonych (comma)
            <input id='plugins_enabled' value='${Array.isArray(plugins.enabled)?plugins.enabled.join(', '):(plugins.enabled||'')}' class='w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm'/>
          </label>
          <button id='savePlugins' class='bg-emerald-600 hover:bg-emerald-500 text-sm px-3 py-1 rounded'>Zapisz Plugins</button>
        </div>
      </div>
    </div>`;
    document.getElementById('refreshCfg').onclick=()=>loadConfig();
    document.getElementById('saveAI').onclick=async()=>{
      const payload = {ai: {default_model: document.getElementById('ai_default_model').value.trim()}};
      const temp = document.getElementById('ai_temperature').value;
      if(temp) payload.ai.temperature = parseFloat(temp);
      try{ await api('/admin/server-config',{method:'PATCH', body: JSON.stringify(payload)}); loadConfig(); }catch(err){ alert(err.message);} };
    document.getElementById('saveTTS').onclick=async()=>{
      const payload = {tts: {engine: document.getElementById('tts_engine').value.trim(), voice: document.getElementById('tts_voice').value.trim()}};
      try{ await api('/admin/server-config',{method:'PATCH', body: JSON.stringify(payload)}); loadConfig(); }catch(err){ alert(err.message);} };
    document.getElementById('savePlugins').onclick=async()=>{
      let raw = document.getElementById('plugins_enabled').value.trim();
      const arr = raw ? raw.split(/[,;]/).map(s=>s.trim()).filter(Boolean) : [];
      const payload = {plugins: {enabled: arr}};
      try{ await api('/admin/server-config',{method:'PATCH', body: JSON.stringify(payload)}); loadConfig(); }catch(err){ alert(err.message);} };
  } catch(e){ c.innerHTML = `<div class='card text-red-400'>Błąd: ${e.message}</div>`; }
}

async function loadLogs(){
  const c = document.getElementById('viewContainer');
  c.innerHTML = '<div class="card">Ładowanie logów...</div>';
  try {
    const logs = await api('/logs?tail=200');
    c.innerHTML = `<div class='card max-h-[70vh] overflow-auto scrollbar-thin'>
      <h3 class='font-semibold mb-3'>Ostatnie logi (${logs.length})</h3>
      <ul class='space-y-1 text-xs font-mono leading-snug'>${logs.map(l=>`<li><span class='text-slate-500'>${l.timestamp}</span> <span class='uppercase ${levelColor(l.level)}'>${l.level}</span> ${escapeHtml(l.message)}</li>`).join('')}</ul>
    </div>`;
  } catch(e){ c.innerHTML = `<div class='card text-red-400'>Błąd: ${e.message}</div>`; }
}

function levelColor(l){
  switch(l){case 'error': return 'text-red-400'; case 'warning': return 'text-amber-400'; case 'debug': return 'text-slate-500'; default: return 'text-emerald-400'; }
}
function escapeHtml(str){ return str.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function loadView(id){
  setNav(id);
  if(id==='dashboard') return loadDashboard();
  if(id==='users') return loadUsers();
  if(id==='devices') return loadDevices();
  if(id==='config') return loadConfig();
  if(id==='logs') return loadLogs();
}

function initAuth(){
  const form = document.getElementById('loginForm');
  form.onsubmit=async(e)=>{
    e.preventDefault();
    const fd = new FormData(form);
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({email: fd.get('email'), password: fd.get('password')})});
      const data = await res.json();
      if(!res.ok || !data.success){ throw new Error(data.detail || data.error || 'Auth failed'); }
      authToken = data.token;
      document.getElementById('loginModal').style.display='none';
      loadView('dashboard');
      pingStatus();
    } catch(err){ document.getElementById('loginError').textContent = err.message; }
  };
  document.getElementById('logoutBtn').onclick=()=>{ authToken=null; document.getElementById('loginModal').style.display='flex'; };
  // Mobile nav toggle
  const mbtn = document.getElementById('mobileMenuBtn');
  if(mbtn){
    mbtn.onclick=()=>{
      const nav = document.getElementById('nav');
      nav.classList.toggle('show-mobile');
      if(nav.classList.contains('show-mobile')){
        nav.classList.remove('hidden');
      } else if(window.innerWidth < 768){
        nav.classList.add('hidden');
      }
    };
    // Hide nav on resize back to desktop
    window.addEventListener('resize',()=>{
      const nav = document.getElementById('nav');
      if(window.innerWidth >= 768){ nav.classList.remove('hidden','show-mobile'); }
      else if(!nav.classList.contains('show-mobile')) nav.classList.add('hidden');
    });
  }
}

async function pingStatus(){
  if(!authToken) return;
  try{
    const metrics = await api('/metrics');
    document.getElementById('uptime').textContent = 'Uptime '+formatDuration(metrics.uptime);
    setTimeout(pingStatus, 10000);
  }catch(e){ setTimeout(pingStatus, 15000); }
}

window.addEventListener('DOMContentLoaded', ()=>{
  setNav('dashboard');
  initAuth();
});
