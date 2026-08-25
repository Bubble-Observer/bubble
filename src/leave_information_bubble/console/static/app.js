(() => {
  'use strict';

  const normalizeAdapters = (value) => {
    const items = Array.isArray(value) ? value : String(value || '').split(',');
    return [...new Set(items.map((item) => String(item).trim()).filter(Boolean))];
  };
  const nonEmptySettings = (value) => Object.fromEntries(
    Object.entries(value || {})
      .map(([key, item]) => [key, String(item ?? '').trim()])
      .filter(([, item]) => item !== ''),
  );
  const isCurrentSelection = (value, epoch, profileId) => (
    value?.selectionEpoch === epoch && value?.profile?.id === profileId
  );
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { isCurrentSelection, normalizeAdapters, nonEmptySettings };
  }
  if (typeof document === 'undefined') return;

  const state = {
    profiles: [],
    profile: null,
    mode: 'broad',
    activeRun: null,
    globalActiveRun: null,
    poller: null,
    editingNewProfile: false,
    cloningProfileId: null,
    memory: null,
    memoryGraph: null,
    memoryGraphWindow: 0,
    memoryGraphWindowCount: 1,
    memoryGraphAutoRotate: true,
    memoryGraphRotationTimer: null,
    memoryGraphHovered: false,
    memoryGraphSelectedId: null,
    memoryObjectId: null,
    pendingWakes: [],
    pendingWakeOutcomes: new Map(),
    workspaceView: 'start',
    selectionEpoch: 0,
    runDrafts: { broad: null, deep: null },
  };
  const MEMORY_GRAPH_ROTATION_MS = 16000;

  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
  })[character]);
  const toList = normalizeAdapters;
  const toCsv = (value) => Array.isArray(value) ? value.join(', ') : (value || '');
  const toLines = (value) => String(value || '').split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
  const branchesToText = (value) => Array.isArray(value)
    ? value.map((branch) => `${branch[0]} :: ${branch[1]}`).join('\n')
    : '';
  const toBranches = (value) => String(value || '').split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line) => {
    const separator = line.indexOf('::');
    if (separator < 1 || separator >= line.length - 2) {
      throw new Error('每个分支必须使用“名称 :: 说明”格式。');
    }
    return [line.slice(0, separator).trim(), line.slice(separator + 2).trim()];
  });
  const profileId = (profile) => profile?.id;
  const runId = (run) => run?.run_id || run?.id;
  const activeStatus = (run) => ['queued', 'running', 'pending'].includes(String(run?.status || '').toLowerCase());
  const display = (value, fallback = '—') => value === null || value === undefined || value === '' ? fallback : value;
  const observerMark = (profile) => {
    const words = String(profile?.display_name || '').match(/[A-Za-z0-9]+/g) || [];
    if (words.length) return words.slice(0, 2).map((word) => word.slice(0, 1)).join('').toUpperCase();
    return String(profile?.domain_key || profile?.id || 'AG')
      .split(/[_-]+/).filter(Boolean).slice(0, 2).map((word) => word.slice(0, 1)).join('').toUpperCase() || 'AG';
  };
  const currentProfileRequest = (epoch, selectedProfileId) => (
    isCurrentSelection(state, epoch, selectedProfileId)
  );

  function syncAdapterChecks(role, value) {
    const selected = new Set(toList(value));
    document.querySelectorAll(`[data-role="${role}"]`).forEach((input) => {
      input.checked = selected.has(input.value);
    });
  }

  function syncAdapterField(role, field) {
    const checkboxes = [...document.querySelectorAll(`[data-role="${role}"]`)];
    const known = new Set(checkboxes.map((input) => input.value));
    const unknown = toList(field.value).filter((value) => !known.has(value));
    field.value = toCsv([
      ...checkboxes.filter((input) => input.checked).map((input) => input.value),
      ...unknown,
    ]);
  }

  function syncReasoningControl(checkbox, select) {
    select.disabled = !checkbox.checked;
    select.setAttribute('aria-disabled', String(!checkbox.checked));
  }

  function defaultRunDraft(profile, mode) {
    const defaults = profile?.defaults || {};
    return {
      perspective: mode === 'deep' ? defaults.deep_perspective || '' : defaults.broad_perspective || '',
      object_id: '',
      max_turns: defaults.max_turns ?? 96,
      max_cost_usd: defaults.max_cost_usd ?? '',
      adapters: toCsv(defaults.adapters),
      thinking: Boolean(defaults.thinking),
      reasoning_effort: defaults.reasoning_effort || '',
    };
  }

  function captureRunDraft(mode = state.mode) {
    const form = $('#run-form');
    if (!form || !state.profile) return;
    state.runDrafts[mode] = {
      perspective: form.elements.perspective.value,
      object_id: form.elements.object_id.value,
      max_turns: form.elements.max_turns.value,
      max_cost_usd: form.elements.max_cost_usd.value,
      adapters: form.elements.adapters.value,
      thinking: form.elements.thinking.checked,
      reasoning_effort: form.elements.reasoning_effort.value,
    };
  }

  function restoreRunDraft(mode) {
    const form = $('#run-form');
    const draft = state.runDrafts[mode] || defaultRunDraft(state.profile, mode);
    form.elements.perspective.value = draft.perspective;
    form.elements.object_id.value = draft.object_id;
    form.elements.max_turns.value = draft.max_turns;
    form.elements.max_cost_usd.value = draft.max_cost_usd;
    form.elements.adapters.value = draft.adapters;
    syncAdapterChecks('run-adapter', draft.adapters);
    form.elements.thinking.checked = draft.thinking;
    form.elements.reasoning_effort.value = draft.reasoning_effort;
    syncReasoningControl(form.elements.thinking, form.elements.reasoning_effort);
  }

  function updateEffectiveDefaults() {
    const form = $('#run-form');
    if (!form) return;
    const sources = toList(form.elements.adapters.value);
    const cost = form.elements.max_cost_usd.value;
    const perspective = form.elements.perspective.value.trim();
    $('#effective-mode').textContent = state.mode === 'deep' ? '深入研究' : '广泛探索';
    $('#effective-turns').textContent = form.elements.max_turns.value || '—';
    $('#effective-cost').textContent = cost ? `$${Number(cost).toFixed(2)}` : '不限制';
    $('#effective-sources').textContent = sources.length ? `${sources.length} 个` : '未选择';
    $('#effective-thinking').textContent = form.elements.thinking.checked
      ? `开启${form.elements.reasoning_effort.value ? ` · ${form.elements.reasoning_effort.value.toUpperCase()}` : ''}`
      : '关闭';
    $('#launch-hint').textContent = perspective
      ? '将从这次临时视角开始；Agent 仍可自行转向。'
      : '无需填写其他内容，可直接由 Agent 自主开始。';
  }

  function setLaunchAvailability() {
    const button = $('#start-run-button');
    const occupied = activeStatus(state.globalActiveRun);
    button.disabled = occupied;
    if (occupied) {
      const sameProfile = state.globalActiveRun?.profile_id === state.profile?.id;
      button.textContent = sameProfile ? '当前 Agent 运行中' : '其他 Agent 运行中';
      $('#launch-hint').textContent = '控制台一次只允许一个运行，结束后会自动恢复。';
    } else {
      button.textContent = state.mode === 'deep' ? '开始深入观察 →' : '开始观察 →';
      updateEffectiveDefaults();
    }
  }

  async function api(url, options = {}) {
    const response = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      let detail = '';
      const raw = await response.text();
      try { detail = JSON.parse(raw).detail || ''; } catch (_) { detail = raw; }
      throw new Error(detail || `请求失败（${response.status}）`);
    }
    return response.status === 204 ? null : response.json();
  }

  function toast(message, error = false) {
    const node = $('#toast');
    node.textContent = message;
    node.className = `toast${error ? ' error' : ''}`;
    node.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { node.hidden = true; }, 4200);
  }

  function openDrawer(drawer) {
    $('#drawer-backdrop').hidden = false;
    drawer.hidden = false;
    document.body.classList.add('drawer-open');
  }

  function closeDrawers() {
    $('#drawer-backdrop').hidden = true;
    $('#settings-drawer').hidden = true;
    $('#connections-drawer').hidden = true;
    $('#prompt-drawer').hidden = true;
    $('#memory-drawer').hidden = true;
    document.body.classList.remove('drawer-open');
  }

  function renderProfiles() {
    const list = $('#agent-list');
    if (!state.profiles.length) {
      list.innerHTML = '<p class="sidebar-muted">还没有 Agent</p>';
      return;
    }
    list.innerHTML = state.profiles.map((profile) => `
      <button type="button" class="agent-item${profileId(profile) === profileId(state.profile) ? ' selected' : ''}" data-profile-id="${escapeHtml(profile.id)}">
        <span><strong>${escapeHtml(profile.display_name || profile.id)}</strong><small>${escapeHtml(profile.domain_key)}</small></span>
      </button>
    `).join('');
    list.querySelectorAll('[data-profile-id]').forEach((button) => {
      button.addEventListener('click', () => selectProfile(button.dataset.profileId));
    });
  }

  function renderLobby() {
    const gallery = $('#agent-gallery');
    $('#agent-count').textContent = String(state.profiles.length);
    const cards = state.profiles.map((profile) => {
      const memory = profile.memory || {};
      const summary = profile.domain_key === 'lol_cn'
        ? '持续观察中文英雄联盟社区，并在必要时纳入国际、跨社区与历史信息。'
        : profile.observation_center;
      return `
        <button class="agent-gallery-card" type="button" data-gallery-profile="${escapeHtml(profile.id)}">
          <span class="gallery-card-top">
            <span class="gallery-avatar" aria-hidden="true">${escapeHtml(observerMark(profile))}</span>
          </span>
          <h2>${escapeHtml(profile.display_name)}</h2>
          <span class="gallery-domain">${escapeHtml(profile.domain_key)}</span>
          <span class="gallery-summary">${escapeHtml(summary)}</span>
          <span class="gallery-stats">
            <span><i class="status-dot"></i>随时可用</span>
            <span>${display(memory.objects, 0)} 个对象</span>
            <span>${memory.commits ? `${display(memory.commits)} 次积累` : '尚未运行'}</span>
          </span>
        </button>
      `;
    }).join('');
    gallery.innerHTML = `${cards}
      <button class="new-agent-card" id="gallery-new-agent" type="button">
        <span class="new-agent-symbol">＋</span><strong>创建新 Agent</strong><span>定义新的领域镜头与独立世界记忆</span>
      </button>`;
    gallery.querySelectorAll('[data-gallery-profile]').forEach((button) => {
      button.addEventListener('click', () => enterWorkspace(button.dataset.galleryProfile));
    });
    $('#gallery-new-agent').addEventListener('click', () => showProfileEditor());
  }

  function showLobby() {
    clearMemoryGraphRotationTimer();
    document.body.classList.add('lobby-mode');
    $('#agent-lobby').hidden = false;
    $('#agent-workspace').hidden = true;
    document.querySelectorAll('.workspace-only').forEach((item) => { item.hidden = true; });
    $('#lobby-nav-button').classList.add('active');
    $('#current-agent-nav-button').classList.remove('active');
    renderLobby();
  }

  async function enterWorkspace(id) {
    const selected = await selectProfile(id);
    if (!selected) return;
    document.body.classList.remove('lobby-mode');
    $('#agent-lobby').hidden = true;
    $('#agent-workspace').hidden = false;
    document.querySelectorAll('.workspace-only').forEach((item) => { item.hidden = false; });
    $('#lobby-nav-button').classList.remove('active');
    $('#current-agent-nav-button').classList.add('active');
    showWorkspacePanel('start');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function showWorkspacePanel(name) {
    const target = ['start', 'memory', 'history'].includes(name) ? name : 'start';
    state.workspaceView = target;
    document.querySelectorAll('[data-workspace-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.workspacePanel !== target;
    });
    document.querySelectorAll('[data-workspace-view]').forEach((button) => {
      const selected = button.dataset.workspaceView === target;
      button.classList.toggle('active', selected);
      if (selected) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    if (target === 'memory') Promise.all([loadMemory(), loadMemoryGraph(), loadPendingWakes()]);
    else clearMemoryGraphRotationTimer();
    if (target === 'history') loadRuns();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function applyProfileToWorkspace(profile) {
    const defaults = profile.defaults || {};
    $('#profile-title').textContent = profile.display_name;
    $('#profile-chip').textContent = profile.domain_key;
    $('#agent-summary-name').textContent = profile.display_name;
    $('#agent-summary-center').textContent = profile.domain_key === 'lol_cn'
      ? '持续观察中文英雄联盟社区，并在必要时纳入国际、跨社区与历史信息。'
      : profile.observation_center;
    $('#agent-summary-locale').textContent = profile.locale;
    $('#agent-summary-timezone').textContent = profile.timezone;
    $('#agent-summary-mode').textContent = defaults.mode === 'deep' ? 'Deep' : 'Broad';

    state.runDrafts = {
      broad: defaultRunDraft(profile, 'broad'),
      deep: defaultRunDraft(profile, 'deep'),
    };
    state.pendingWakes = [];
    state.pendingWakeOutcomes = new Map();
    setMode(defaults.mode || 'broad', { reloadPrompt: false, initialize: true });
  }

  function fillProfileEditor(profile) {
    const form = $('#profile-form');
    const defaults = profile.defaults || {};
    ['display_name', 'domain_key', 'observation_center', 'relevance_rule', 'locale', 'timezone', 'world_db', 'runtime_db'].forEach((name) => {
      form.elements[name].value = profile[name] ?? '';
    });
    form.elements.id.value = profile.id;
    form.elements.id.readOnly = true;
    form.elements.attention_examples.value = toCsv(profile.attention_examples);
    form.elements.source_preferences.value = toCsv(profile.source_preferences);
    form.elements.branches.value = branchesToText(profile.branches);
    form.elements.search_experience.value = (profile.search_experience || []).join('\n');
    form.elements.default_broad_perspective.value = defaults.broad_perspective || '';
    form.elements.default_deep_perspective.value = defaults.deep_perspective || '';
    form.elements.default_mode.value = defaults.mode || 'broad';
    form.elements.default_adapters.value = toCsv(defaults.adapters);
    syncAdapterChecks('profile-adapter', defaults.adapters);
    form.elements.default_max_turns.value = defaults.max_turns ?? 96;
    form.elements.default_max_cost_usd.value = defaults.max_cost_usd ?? '';
    form.elements.default_thinking.checked = Boolean(defaults.thinking);
    form.elements.default_reasoning_effort.value = defaults.reasoning_effort || '';
    syncReasoningControl(
      form.elements.default_thinking,
      form.elements.default_reasoning_effort,
    );
  }

  function newProfileTemplate() {
    return {
      id: '', display_name: '', domain_key: '', observation_center: '', relevance_rule: '',
      attention_examples: [], source_preferences: [], branches: [], search_experience: [], locale: 'zh-CN', timezone: 'Asia/Shanghai',
      world_db: '', runtime_db: '',
      defaults: {
        mode: 'broad', adapters: ['bilibili', 'public-web'], max_turns: 96,
        max_cost_usd: null, thinking: false, reasoning_effort: null,
        broad_perspective: null,
        deep_perspective: null,
      },
    };
  }

  function showProfileEditor(profile = null) {
    state.editingNewProfile = !profile;
    state.cloningProfileId = null;
    $('#settings-title').textContent = profile ? `编辑 ${profile.display_name}` : '新建 Agent';
    $('#profile-form').reset();
    $('#profile-form').classList.toggle('quick-create', !profile);
    $('#profile-form').classList.remove('clone-create');
    $('#clone-profile-button').hidden = !profile;
    fillProfileEditor(profile || newProfileTemplate());
    $('#profile-form').elements.id.readOnly = Boolean(profile);
    const experts = [...$('#profile-form').querySelectorAll('.expert-settings')];
    experts.forEach((expert) => {
      expert.open = false;
      expert.classList.toggle('deferred', !profile);
    });
    $('#expert-settings-hint').textContent = profile
      ? '数据库、预算、领域规则与诊断能力'
      : '创建后可继续设置数据库、预算与领域规则';
    openDrawer($('#settings-drawer'));
  }

  async function selectProfile(id) {
    const epoch = ++state.selectionEpoch;
    try {
      const profile = await api(`/api/profiles/${encodeURIComponent(id)}`);
      if (epoch !== state.selectionEpoch) return false;
      state.profile = profile;
      state.memoryGraphWindow = 0;
      state.memoryGraphWindowCount = 1;
      state.memoryGraphSelectedId = null;
      clearMemoryGraphRotationTimer();
      $('#console-content').hidden = false;
      $('#empty-profile').hidden = true;
      renderProfiles();
      applyProfileToWorkspace(state.profile);
      await Promise.all([
        loadPrompt(epoch, id),
        loadRuns(epoch, id),
        loadMemory(epoch, id),
        loadMemoryGraph(epoch, id),
        loadPendingWakes(epoch, id),
      ]);
      return currentProfileRequest(epoch, id);
    } catch (error) {
      if (epoch === state.selectionEpoch) toast(error.message, true);
      return false;
    }
  }

  async function loadProfiles(preferredId = null) {
    try {
      const result = await api('/api/profiles');
      state.profiles = Array.isArray(result) ? result : (result.items || result.profiles || []);
      renderProfiles();
      renderLobby();
      if (preferredId) await enterWorkspace(preferredId);
    } catch (error) {
      $('#agent-list').innerHTML = '<p class="sidebar-muted">无法读取 Agent</p>';
      toast(error.message, true);
    }
  }

  async function saveProfile(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const raw = Object.fromEntries(new FormData(form).entries());
    const operatorInstructions = state.editingNewProfile ? null : state.profile?.operator_instructions || null;
    try {
      if (!toList(raw.default_adapters).length) {
        throw new Error('请至少选择一个默认信息来源。');
      }
      if (state.cloningProfileId) {
        const cloned = await api(`/api/profiles/${encodeURIComponent(state.cloningProfileId)}/clone`, {
          method: 'POST',
          body: JSON.stringify({ id: raw.id || null, display_name: raw.display_name }),
        });
        toast('领域设计已复制；新 Agent 的记忆为空，将在第一次 Wake 时建库。');
        closeDrawers();
        await loadProfiles();
        await enterWorkspace(cloned.id);
        return;
      }
      if (state.editingNewProfile) {
        const created = await api('/api/profiles/quick', {
          method: 'POST',
          body: JSON.stringify({
            id: raw.id || null,
            display_name: raw.display_name,
            observation_center: raw.observation_center,
            locale: raw.locale,
            timezone: raw.timezone,
            adapters: toList(raw.default_adapters),
          }),
        });
        toast('Agent 已创建；数据库会在第一次 Wake 时自动生成。');
        closeDrawers();
        await loadProfiles();
        await enterWorkspace(created.id);
        return;
      }
      const defaultDirectory = `data/agents/${raw.id}`;
      const profile = {
        id: raw.id,
        display_name: raw.display_name,
        domain_key: raw.domain_key,
        observation_center: raw.observation_center,
        relevance_rule: raw.relevance_rule,
        attention_examples: toList(raw.attention_examples),
        source_preferences: toList(raw.source_preferences),
        branches: toBranches(raw.branches),
        search_experience: toLines(raw.search_experience),
        locale: raw.locale,
        timezone: raw.timezone,
        world_db: raw.world_db || `${defaultDirectory}/world.sqlite3`,
        runtime_db: raw.runtime_db || `${defaultDirectory}/runtime.sqlite3`,
        operator_instructions: operatorInstructions,
        defaults: {
          mode: raw.default_mode,
          broad_perspective: raw.default_broad_perspective || null,
          deep_perspective: raw.default_deep_perspective || null,
          adapters: toList(raw.default_adapters),
          max_turns: Number(raw.default_max_turns),
          max_cost_usd: raw.default_max_cost_usd === '' ? null : Number(raw.default_max_cost_usd),
          thinking: form.elements.default_thinking.checked,
          reasoning_effort: form.elements.default_thinking.checked
            ? raw.default_reasoning_effort || null
            : null,
        },
      };
      await api(`/api/profiles/${encodeURIComponent(profile.id)}`, {
        method: 'PUT',
        body: JSON.stringify(profile),
      });
      toast('Agent 配置已保存。');
      closeDrawers();
      await loadProfiles();
      await enterWorkspace(profile.id);
    } catch (error) { toast(error.message, true); }
  }

  function showCloneEditor() {
    if (!state.profile) return;
    state.editingNewProfile = true;
    state.cloningProfileId = state.profile.id;
    $('#settings-title').textContent = `复制 ${state.profile.display_name}`;
    $('#profile-form').reset();
    fillProfileEditor(state.profile);
    $('#profile-form').elements.id.value = '';
    $('#profile-form').elements.id.readOnly = false;
    $('#profile-form').elements.display_name.value = `${state.profile.display_name} 副本`;
    $('#profile-form').classList.add('quick-create', 'clone-create');
    $('#profile-form').querySelectorAll('.expert-settings').forEach((expert) => expert.classList.add('deferred'));
    $('#expert-settings-hint').textContent = '完整专业配置会从原 Agent 复制';
    $('#clone-profile-button').hidden = true;
  }

  function currentDefaultPerspective() {
    if (!state.profile) return '';
    return state.mode === 'deep'
      ? state.profile.defaults?.deep_perspective || ''
      : state.profile.defaults?.broad_perspective || '';
  }

  function resetPerspective() {
    $('#run-form').elements.perspective.value = currentDefaultPerspective();
    captureRunDraft();
    updateEffectiveDefaults();
  }

  function setMode(mode, options = {}) {
    const { reloadPrompt = true, initialize = false } = options;
    if (!initialize && state.profile) captureRunDraft(state.mode);
    state.mode = mode;
    document.querySelectorAll('.mode-card').forEach((card) => {
      const selected = card.dataset.mode === mode;
      card.classList.toggle('selected', selected);
      const input = card.querySelector('input[type="radio"]');
      if (input) input.checked = selected;
    });
    $('#wake-options').hidden = mode !== 'deep';
    $('#prompt-mode-title').textContent = mode === 'deep' ? 'Deep Prompt' : 'Broad Prompt';
    $('#prompt-drawer-mode').textContent = mode === 'deep' ? 'Deep · 深度探索' : 'Broad · 广域扫描';
    restoreRunDraft(mode);
    updateEffectiveDefaults();
    setLaunchAvailability();
    if (reloadPrompt) loadPrompt();
  }

  function renderPrompt(data) {
    const layerDefinitions = [
      ['stable', '稳定身份', '核心层 · 只读'],
      ['domain', '领域镜头', '来自 Agent 配置'],
      ['posture', state.mode === 'deep' ? 'Deep 工作姿态' : 'Broad 工作姿态', '随模式切换'],
      ['epistemic', '认识论约束', '核心层 · 只读'],
      ['mechanics', '运行机制', '随 Wake 协议切换'],
      ['operator', '操作员长期引导', '高级设置 · 边界内生效'],
    ].filter(([key]) => data[key] !== null && data[key] !== undefined);

    $('#prompt-layers').innerHTML = layerDefinitions.map(([key, title, badge]) => `
      <div class="prompt-layer">
        <strong>${escapeHtml(title)}</strong>
        <span>${escapeHtml(badge)} · ${escapeHtml(String(data[key]).replace(/\s+/g, ' ').slice(0, 120))}</span>
      </div>
    `).join('');
    $('#compiled-prompt').textContent = data.compiled || '暂无编译结果。';
    $('#operator-instructions').value = data.operator ?? state.profile?.operator_instructions ?? '';
    $('#wake-input-preview').textContent = data.wake_input || '—';
    const form = $('#run-form');
    const seed = state.mode === 'deep' ? form.elements.object_id.value.trim() : '';
    const cost = form.elements.max_cost_usd.value;
    $('#wake-preview-mode').textContent = state.mode === 'deep' ? 'Deep' : 'Broad';
    $('#wake-preview-seed').textContent = seed || 'Agent 自主选择';
    $('#wake-preview-budget').textContent = `${form.elements.max_turns.value} 轮 · ${cost ? `$${Number(cost).toFixed(2)}` : '不另设成本上限'}`;
    $('#wake-preview-adapters').textContent = form.elements.adapters.value;
    $('#prompt-summary-text').textContent = `系统 Prompt 由 ${layerDefinitions.length} 层组合；本次 Wake 输入单独显示，不会混入核心规则。`;
  }

  async function loadPrompt(epoch = state.selectionEpoch, selectedProfileId = state.profile?.id) {
    if (!selectedProfileId) return;
    try {
      const data = await api(`/api/profiles/${encodeURIComponent(selectedProfileId)}/prompt-preview`, {
        method: 'POST',
        body: JSON.stringify({
          mode: state.mode,
          object_id: state.mode === 'deep' ? $('#run-form').elements.object_id.value || null : null,
          perspective: $('#run-form').elements.perspective.value || null,
        }),
      });
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      renderPrompt(data);
    } catch (error) {
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      $('#prompt-summary-text').textContent = `无法组合 Prompt：${error.message}`;
    }
  }

  async function saveOperator() {
    if (!state.profile) return;
    try {
      state.profile = await api(`/api/profiles/${encodeURIComponent(state.profile.id)}`, {
        method: 'PUT',
        body: JSON.stringify({ ...state.profile, operator_instructions: $('#operator-instructions').value }),
      });
      toast('系统 Prompt 附加视角已保存。');
      await loadPrompt();
    } catch (error) { toast(error.message, true); }
  }

  function renderMemory(data, query = '') {
    state.memory = data;
    const summary = data?.summary || {};
    $('#memory-summary').innerHTML = [
      ['对象', summary.objects || 0],
      ['当前判断', summary.current_assertions || 0],
      ['开放疑问', summary.open_inquiries || 0],
      ['认知提交', summary.cognition_commits || 0],
    ].map(([label, value]) => `<span class="metric"><strong>${escapeHtml(value)}</strong><small>${label}</small></span>`).join('');
    renderMemoryObjects(data?.object_samples || data?.recent_objects || [], query, !query);
    renderMemoryInquiries(data?.open_inquiries || []);
    updateObjectSeedOptions(data?.object_samples || data?.recent_objects || []);
  }

  function renderMemoryObjects(items, query = '', isSample = false) {
    const root = $('#memory-object-list');
    if (!items.length) {
      root.innerHTML = `<div class="empty-state"><strong>${query ? '没有匹配对象' : '还没有持久化对象'}</strong><span>${query ? '可换一个名称或别名。' : '完成一次产生认知写入的 Wake 后会出现在这里。'}</span></div>`;
      return;
    }
    root.innerHTML = `${isSample ? '<p class="form-help">对象入口（有界，按名称排序；不是最近对象）。</p>' : ''}` + items.map((item) => `
      <button type="button" class="memory-object" data-memory-object="${escapeHtml(item.id)}">
        <span><strong>${escapeHtml(item.canonical_name)}</strong><small>${escapeHtml(item.kind)} · ${display(item.assertion_count, 0)} 条当前判断</small></span>
        <span class="memory-chevron">查看 →</span>
      </button>
    `).join('');
    root.querySelectorAll('[data-memory-object]').forEach((button) => {
      button.addEventListener('click', () => openMemoryObject(button.dataset.memoryObject));
    });
  }

  function renderMemoryInquiries(items) {
    $('#memory-inquiry-list').innerHTML = items.length ? items.map((item) => `
      <div class="memory-inquiry"><strong>${escapeHtml(item.prompt)}</strong><span>${escapeHtml(item.subject?.name || item.subject?.id || '')}</span></div>
    `).join('') : '<p class="form-help">当前没有开放疑问。</p>';
  }

  function updateObjectSeedOptions(items) {
    const select = $('#run-form').elements.object_id;
    const current = select.value || state.runDrafts.deep?.object_id || '';
    const unique = new Map(items.map((item) => [item.id, item]));
    if (current && !unique.has(current)) unique.set(current, { id: current, canonical_name: current });
    select.innerHTML = '<option value="">由 Agent 自主选择</option>' + Array.from(unique.values()).map((item) => `
      <option value="${escapeHtml(item.id)}">${escapeHtml(item.canonical_name)}</option>
    `).join('');
    select.value = current;
  }

  async function loadMemory(epoch = state.selectionEpoch, selectedProfileId = state.profile?.id) {
    if (!selectedProfileId) return;
    try {
      const data = await api(`/api/profiles/${encodeURIComponent(selectedProfileId)}/memory`);
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      renderMemory(data);
    } catch (error) {
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      $('#memory-object-list').innerHTML = `<div class="empty-state"><strong>无法读取认知库</strong><span>${escapeHtml(error.message)}</span></div>`;
    }
  }

  function graphNodeDegrees(nodes, edges) {
    const degrees = new Map(nodes.map((node) => [node.id, 0]));
    edges.forEach((edge) => {
      degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1);
      if (edge.target !== edge.source) degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1);
    });
    return degrees;
  }

  function graphNodeRadius(node, degree) {
    const activity = Math.min(Math.sqrt(Number(node.assertion_count || 0)), 4.5);
    return 17 + activity * 1.7 + Math.min(degree, 6) * 1.15;
  }

  function graphPositions(nodes, edges) {
    const center = { x: 460, y: 268 };
    const degrees = graphNodeDegrees(nodes, edges);
    if (nodes.length <= 1) {
      return nodes.map((node) => ({ ...center, radius: graphNodeRadius(node, degrees.get(node.id) || 0) }));
    }
    if (!edges.length) {
      const columns = Math.min(6, Math.ceil(Math.sqrt(nodes.length * 1.55)));
      const rows = Math.ceil(nodes.length / columns);
      return nodes.map((node, index) => ({
        x: 100 + (index % columns) * (720 / Math.max(columns - 1, 1)),
        y: 130 + Math.floor(index / columns) * (285 / Math.max(rows - 1, 1)),
        radius: graphNodeRadius(node, 0),
      }));
    }

    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    const positions = nodes.map((node, index) => {
      const angle = index * goldenAngle;
      const distance = Math.min(52 + Math.sqrt(index) * 58, 285);
      return {
        x: center.x + Math.cos(angle) * distance,
        y: center.y + Math.sin(angle) * distance * 0.65,
        vx: 0,
        vy: 0,
        radius: graphNodeRadius(node, degrees.get(node.id) || 0),
      };
    });
    const indexes = new Map(nodes.map((node, index) => [node.id, index]));
    for (let iteration = 0; iteration < 150; iteration += 1) {
      const alpha = 1 - iteration / 150;
      const forces = positions.map(() => ({ x: 0, y: 0 }));
      for (let left = 0; left < positions.length; left += 1) {
        for (let right = left + 1; right < positions.length; right += 1) {
          const dx = positions[right].x - positions[left].x;
          const dy = positions[right].y - positions[left].y;
          const distance = Math.max(Math.hypot(dx, dy), 1);
          const minimum = positions[left].radius + positions[right].radius + 44;
          const repulsion = (2600 / (distance * distance) + Math.max(minimum - distance, 0) * 0.018) * alpha;
          const fx = (dx / distance) * repulsion;
          const fy = (dy / distance) * repulsion;
          forces[left].x -= fx;
          forces[left].y -= fy;
          forces[right].x += fx;
          forces[right].y += fy;
        }
      }
      edges.forEach((edge) => {
        const sourceIndex = indexes.get(edge.source);
        const targetIndex = indexes.get(edge.target);
        if (sourceIndex === undefined || targetIndex === undefined || sourceIndex === targetIndex) return;
        const source = positions[sourceIndex];
        const target = positions[targetIndex];
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const distance = Math.max(Math.hypot(dx, dy), 1);
        const desired = 112 + Math.min(source.radius + target.radius, 48);
        const pull = (distance - desired) * 0.012 * alpha;
        const fx = (dx / distance) * pull;
        const fy = (dy / distance) * pull;
        forces[sourceIndex].x += fx;
        forces[sourceIndex].y += fy;
        forces[targetIndex].x -= fx;
        forces[targetIndex].y -= fy;
      });
      positions.forEach((position, index) => {
        forces[index].x += (center.x - position.x) * 0.0025 * alpha;
        forces[index].y += (center.y - position.y) * 0.0035 * alpha;
        position.vx = (position.vx + forces[index].x) * 0.76;
        position.vy = (position.vy + forces[index].y) * 0.76;
        position.x = Math.min(858, Math.max(62, position.x + position.vx));
        position.y = Math.min(444, Math.max(92, position.y + position.vy));
      });
    }
    return positions;
  }

  function graphEdgePath(source, target, offset = 0) {
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const distance = Math.max(Math.hypot(dx, dy), 1);
    if (distance <= 2) {
      const radius = source.radius + 18;
      return `M ${source.x} ${source.y - source.radius} C ${source.x + radius * 1.4} ${source.y - radius * 1.8}, ${source.x + radius * 1.4} ${source.y + radius * 1.8}, ${source.x} ${source.y + source.radius}`;
    }
    const ux = dx / distance;
    const uy = dy / distance;
    const startX = source.x + ux * (source.radius + 4);
    const startY = source.y + uy * (source.radius + 4);
    const endX = target.x - ux * (target.radius + 4);
    const endY = target.y - uy * (target.radius + 4);
    const middleX = (startX + endX) / 2 - uy * offset;
    const middleY = (startY + endY) / 2 + ux * offset;
    return `M ${startX} ${startY} Q ${middleX} ${middleY} ${endX} ${endY}`;
  }

  function compactDatabasePath(value) {
    const parts = String(value || '').replaceAll('\\', '/').split('/').filter(Boolean);
    return parts.length ? parts.slice(-3).join('/') : '尚未选择数据库';
  }

  function formatLocalTime(value) {
    // 后端时间以 aware UTC（+00:00）存储；这里统一按浏览器本地时区渲染，
    // 避免把 ISO 字符串原样显示成比本地慢 8 小时的 UTC 时刻。
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('zh-CN', {
      year: 'numeric', month: 'numeric', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    }).format(date);
  }

  function graphWrittenAt(value) {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('zh-CN', {
      month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false,
    }).format(date);
  }

  function clearMemoryGraphRotationTimer() {
    if (state.memoryGraphRotationTimer) clearTimeout(state.memoryGraphRotationTimer);
    state.memoryGraphRotationTimer = null;
  }

  function updateMemoryGraphRotationControls(snapshot) {
    const controls = $('#memory-graph-window-controls');
    const stage = controls.closest('.graph-stage');
    const windowState = snapshot?.window || {};
    state.memoryGraphWindow = Number(windowState.index || 0);
    state.memoryGraphWindowCount = Math.max(1, Number(windowState.count || 1));
    const rotating = Boolean(windowState.rotating) && state.memoryGraphWindowCount > 1;
    controls.hidden = !rotating;
    stage.classList.toggle('has-rotation', rotating);
    $('#memory-graph-window-label').textContent = `${state.memoryGraphWindow + 1} / ${state.memoryGraphWindowCount}`;
    const autoButton = $('#memory-graph-auto');
    autoButton.classList.toggle('active', state.memoryGraphAutoRotate);
    autoButton.setAttribute('aria-pressed', String(state.memoryGraphAutoRotate));
  }

  function scheduleMemoryGraphRotation() {
    clearMemoryGraphRotationTimer();
    if (!state.memoryGraphAutoRotate
      || state.memoryGraphWindowCount <= 1
      || state.workspaceView !== 'memory'
      || state.memoryGraphHovered
      || state.memoryGraphSelectedId
      || document.hidden) return;
    state.memoryGraphRotationTimer = setTimeout(() => rotateMemoryGraph(1), MEMORY_GRAPH_ROTATION_MS);
  }

  async function rotateMemoryGraph(direction) {
    if (state.memoryGraphWindowCount <= 1 || !state.profile?.id) return;
    clearMemoryGraphRotationTimer();
    const epoch = state.selectionEpoch;
    const profileId = state.profile.id;
    const nextWindow = (
      state.memoryGraphWindow + direction + state.memoryGraphWindowCount
    ) % state.memoryGraphWindowCount;
    const stage = $('.graph-stage');
    stage.classList.add('is-rotating');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    await new Promise((resolve) => setTimeout(resolve, reducedMotion ? 0 : 140));
    state.memoryGraphWindow = nextWindow;
    state.memoryGraphSelectedId = null;
    await loadMemoryGraph(epoch, profileId, nextWindow);
    stage.classList.remove('is-rotating');
  }

  function renderGraphStatus(snapshot) {
    const sourceRoot = $('#memory-graph-source');
    const statsRoot = $('#memory-graph-stats');
    if (!snapshot?.available) {
      sourceRoot.textContent = '数据库尚未建立';
      sourceRoot.removeAttribute('title');
      statsRoot.innerHTML = '<span><strong>—</strong><small>对象</small></span><span><strong>—</strong><small>关系</small></span>';
      return;
    }
    const stats = snapshot.stats || {};
    const commitTime = graphWrittenAt(snapshot.latest_commit?.committed_at);
    const schema = snapshot.schema_version ? ` · schema v${snapshot.schema_version}` : '';
    const displayedConnections = display(stats.displayed_connections, stats.displayed_relations || 0);
    const totalConnections = display(stats.total_object_connections, stats.total_object_relations || 0);
    sourceRoot.textContent = `${compactDatabasePath(state.profile?.world_db)}${schema}${commitTime ? ` · 最近发布 ${commitTime}` : ''}`;
    sourceRoot.title = [state.profile?.world_db, snapshot.latest_commit?.commit_id].filter(Boolean).join(' · ');
    statsRoot.innerHTML = `
      <span><strong>${escapeHtml(display(stats.displayed_nodes, 0))} / ${escapeHtml(display(stats.total_nodes, 0))}</strong><small>视图 / 全库对象</small></span>
      <span><strong>${escapeHtml(displayedConnections)} / ${escapeHtml(totalConnections)}</strong><small>视图 / 全库连接</small></span>`;
  }

  function renderGraphDetail(node) {
    state.memoryGraphSelectedId = node?.id || null;
    scheduleMemoryGraphRotation();
    const root = $('#memory-graph-detail');
    root.classList.toggle('has-selection', Boolean(node));
    root.scrollTop = 0;
    const snapshot = state.memoryGraph || {};
    const edges = snapshot.edges || [];
    const nodesById = new Map((snapshot.nodes || []).map((item) => [item.id, item]));
    document.querySelectorAll('#memory-graph .graph-node').forEach((item) => {
      item.classList.toggle('selected', Boolean(node) && item.dataset.objectId === node.id);
      item.classList.toggle('muted', Boolean(node) && item.dataset.objectId !== node.id
        && !edges.some((edge) => (edge.source === node.id && edge.target === item.dataset.objectId)
          || (edge.target === node.id && edge.source === item.dataset.objectId)));
    });
    document.querySelectorAll('#memory-graph .graph-edge').forEach((item) => {
      const connected = Boolean(node)
        && (item.dataset.source === node.id || item.dataset.target === node.id);
      item.classList.toggle('connected', connected);
      item.classList.toggle('muted', Boolean(node) && !connected);
    });
    if (!node) {
      root.innerHTML = '<span class="graph-detail-orbit" aria-hidden="true">◎</span><p class="eyebrow">How to read</p><h3>圆点是对象，连线是关联</h3><p>连接可以从任一端浏览。点击节点查看它与谁相连；文字属性保留在对象详情中。</p>';
      return;
    }
    const relations = edges.filter((edge) => edge.source === node.id || edge.target === node.id);
    const relationItems = relations.slice(0, 5).map((edge) => {
      const assertionCount = Number(edge.assertion_count || 1);
      const countLabel = assertionCount > 1 ? `（${assertionCount} 条当前判断）` : '';
      const peerId = edge.source === node.id ? edge.target : edge.source;
      const peer = nodesById.get(peerId);
      return `<li><b>${escapeHtml(edge.predicate || '相关')}</b><span aria-hidden="true"> · </span>${escapeHtml(peer?.canonical_name || peerId)}${escapeHtml(countLabel)}</li>`;
    }).join('');
    const moreRelations = relations.length > 5 ? `<li>还有 ${relations.length - 5} 条关系未在此展开</li>` : '';
    root.innerHTML = `
      <span class="graph-detail-orbit" aria-hidden="true">✦</span>
      <p class="eyebrow">${escapeHtml(node.kind || 'object')}</p>
      <h3>${escapeHtml(node.canonical_name || node.id)}</h3>
      <p>${escapeHtml(node.provisional ? '这是一个仍在确认中的临时对象。' : '这是持久化数据库中的正式对象。')}</p>
      <dl>
        <dt>当前判断</dt><dd>${escapeHtml(display(node.assertion_count, 0))} 条</dd>
        <dt>画面内关系</dt><dd>${relations.length} 条</dd>
        <dt>开放问题</dt><dd>${escapeHtml(display(node.inquiry_count, 0))} 条</dd>
      </dl>
      ${relationItems ? `<ul class="graph-relations">${relationItems}${moreRelations}</ul>` : '<p>这个对象在当前视图中没有对象关系；它仍可能有文字属性。</p>'}
      <button class="button ghost" type="button" data-open-graph-object="${escapeHtml(node.id)}">查看完整记忆</button>`;
    root.querySelector('[data-open-graph-object]').addEventListener('click', () => openMemoryObject(node.id));
  }

  function renderMemoryGraph(snapshot) {
    state.memoryGraph = snapshot;
    const nodeRoot = $('#memory-graph-nodes');
    const edgeRoot = $('#memory-graph-edges');
    nodeRoot.replaceChildren();
    edgeRoot.replaceChildren();
    const nodes = snapshot?.nodes || [];
    const edges = snapshot?.edges || [];
    const caption = $('#memory-graph-caption');
    renderGraphStatus(snapshot);
    updateMemoryGraphRotationControls(snapshot);
    renderGraphDetail(null);
    const renderPlaceholder = () => {
      const namespace = 'http://www.w3.org/2000/svg';
      const group = document.createElementNS(namespace, 'g');
      group.setAttribute('class', 'graph-placeholder');
      group.setAttribute('transform', 'translate(450 245)');
      [0, 60, 120].forEach((rotation, index) => {
        const orbit = document.createElementNS(namespace, 'ellipse');
        orbit.setAttribute('rx', '78');
        orbit.setAttribute('ry', '28');
        orbit.setAttribute('transform', `rotate(${rotation})`);
        orbit.setAttribute('class', `placeholder-orbit orbit-${index + 1}`);
        group.append(orbit);
      });
      const center = document.createElementNS(namespace, 'circle');
      center.setAttribute('r', '9');
      center.setAttribute('class', 'placeholder-center');
      group.append(center);
      nodeRoot.append(group);
    };
    if (!snapshot?.available) {
      renderPlaceholder();
      caption.textContent = '数据库尚未建立，完成第一次有效探索后会出现图谱。';
      return;
    }
    if (!nodes.length) {
      renderPlaceholder();
      caption.textContent = '数据库已连接，当前还没有持久化对象。';
      return;
    }
    const namespace = 'http://www.w3.org/2000/svg';
    const positions = graphPositions(nodes, edges);
    const byId = new Map(nodes.map((node, index) => [node.id, { node, position: positions[index] }]));
    const pairIndexes = new Map();
    edges.forEach((edge) => {
      const sourceEntry = byId.get(edge.source);
      const targetEntry = byId.get(edge.target);
      const source = sourceEntry?.position;
      const target = targetEntry?.position;
      if (!source || !target) return;
      const pairKey = [edge.source, edge.target].sort().join('\u0000');
      const pairIndex = pairIndexes.get(pairKey) || 0;
      pairIndexes.set(pairKey, pairIndex + 1);
      const offset = pairIndex === 0 ? 0 : Math.ceil(pairIndex / 2) * 18 * (pairIndex % 2 ? 1 : -1);
      const path = document.createElementNS(namespace, 'path');
      path.setAttribute('class', 'graph-edge');
      path.setAttribute('d', graphEdgePath(source, target, offset));
      path.dataset.source = edge.source;
      path.dataset.target = edge.target;
      const title = document.createElementNS(namespace, 'title');
      const assertionCount = Number(edge.assertion_count || 1);
      title.textContent = `${sourceEntry.node.canonical_name || edge.source} — ${edge.predicate || '相关'} — ${targetEntry.node.canonical_name || edge.target}${assertionCount > 1 ? `（聚合 ${assertionCount} 条当前判断）` : ''}`;
      path.append(title);
      edgeRoot.append(path);
    });
    const palette = ['#656bc6', '#70aa9b', '#d89077', '#d2ae61', '#8c83b9'];
    const degrees = graphNodeDegrees(nodes, edges);
    nodes.forEach((node, index) => {
      const position = positions[index];
      const group = document.createElementNS(namespace, 'g');
      group.setAttribute('class', 'graph-node');
      group.setAttribute('transform', `translate(${position.x} ${position.y})`);
      group.setAttribute('tabindex', '0');
      group.setAttribute('role', 'button');
      group.style.setProperty('--node-delay', `${Math.min(index, 12) * 18}ms`);
      const degree = degrees.get(node.id) || 0;
      group.setAttribute('aria-label', `${node.canonical_name || node.id}，${node.assertion_count || 0} 条当前判断，${degree} 条画面内关系`);
      group.dataset.objectId = node.id;
      const circle = document.createElementNS(namespace, 'circle');
      circle.setAttribute('r', String(position.radius));
      const kind = String(node.kind || 'object');
      const colorIndex = Array.from(kind).reduce((total, character) => total + character.charCodeAt(0), 0) % palette.length;
      circle.setAttribute('fill', palette[colorIndex]);
      const label = document.createElementNS(namespace, 'text');
      label.setAttribute('y', String(position.radius + 18));
      const name = String(node.canonical_name || node.id);
      const characters = Array.from(name);
      label.textContent = characters.length > 12 ? `${characters.slice(0, 11).join('')}…` : name;
      const title = document.createElementNS(namespace, 'title');
      title.textContent = `${name} · ${degree} 条画面内关系`;
      group.append(circle, label, title);
      const selectNode = () => renderGraphDetail(node);
      group.addEventListener('click', selectNode);
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectNode();
        }
      });
      nodeRoot.append(group);
    });
    const stats = snapshot.stats || {};
    const literalCount = Number(stats.total_literal_assertions || 0);
    const collapsedCount = Number(stats.collapsed_relations || 0);
    if (!edges.length && literalCount) {
      caption.textContent = `这个库有 ${literalCount} 条文字属性，但没有对象到对象的关系；属性可在节点详情查看，因此这里不会强行画线。`;
    } else {
      const bounded = snapshot.truncated
        ? `当前显示 ${display(stats.displayed_nodes, nodes.length)} / ${display(stats.total_nodes, nodes.length)} 个对象的局部视图。`
        : '当前对象已全部显示。';
      const aggregation = collapsedCount ? `同语义平行关系已聚合，当前少画 ${collapsedCount} 条重复线。` : '';
      const rotation = state.memoryGraphWindowCount > 1 ? '闲置时会轮换记忆视窗，悬停或选中节点即暂停。' : '';
      caption.textContent = `优先展示有关系的对象；连线表示可从任一端浏览的关联。${literalCount} 条文字属性不画成节点。${aggregation}${bounded}${rotation}`;
    }
  }

  async function loadMemoryGraph(
    epoch = state.selectionEpoch,
    selectedProfileId = state.profile?.id,
    windowIndex = state.memoryGraphWindow,
  ) {
    if (!selectedProfileId) return;
    try {
      const snapshot = await api(`/api/profiles/${encodeURIComponent(selectedProfileId)}/memory/graph?limit=24&window=${encodeURIComponent(windowIndex)}`);
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      renderMemoryGraph(snapshot);
    } catch (error) {
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      renderMemoryGraph({ available: false, nodes: [], edges: [], truncated: false });
      $('#memory-graph-caption').textContent = `无法读取图谱：${error.message}`;
    }
  }

  function pendingOutcomeText(outcome) {
    if (!outcome) return '';
    const statusNames = {
      published: '已发布到正式记忆',
      already_published: '此前已经发布',
      blocked: '暂时无法发布',
      compile_failed: '暂存内容校验失败',
      commit_rejected: '正式记忆拒绝了这次写入',
      nothing_to_finalize: '没有可发布的内容',
      error: '发布请求失败',
    };
    const problems = Array.isArray(outcome.problems) ? outcome.problems : [];
    const blockers = (Array.isArray(outcome.blockers) ? outcome.blockers : []).map((entry) => {
      if (typeof entry === 'string') return entry;
      return entry.message || entry.action_hint || [entry.code, entry.ref].filter(Boolean).join(' · ');
    }).filter(Boolean);
    const details = [...problems, ...blockers].join('；');
    return `${statusNames[outcome.status] || outcome.status || '发布结果'}${details ? `：${details}` : ''}`;
  }

  function renderPendingWakes(wakes) {
    const root = $('#pending-wakes-list');
    if (!wakes.length) {
      root.innerHTML = '<p class="pending-empty">没有待发布内容，正式记忆与图谱已是当前状态。</p>';
      return;
    }
    root.innerHTML = wakes.map((wake) => {
      const staging = wake.staging || {};
      const outcome = state.pendingWakeOutcomes.get(wake.wake_id);
      const result = pendingOutcomeText(outcome);
      const error = outcome && !['published', 'already_published', 'nothing_to_finalize'].includes(outcome.status);
      return `
        <article class="pending-wake">
          <div class="pending-wake-copy">
            <strong title="${escapeHtml(wake.wake_id)}">${escapeHtml(wake.wake_id)}</strong>
            <span>暂存 ${escapeHtml(display(wake.staging_total, 0))} 项 · 对象 ${escapeHtml(display(staging.staged_objects, 0))} · 判断 ${escapeHtml(display(staging.staged_assertions, 0))} · 问题 ${escapeHtml(display(staging.staged_inquiries, 0))}</span>
            <span>来源运行：${escapeHtml(display(wake.claimed_by, '未知'))}</span>
          </div>
          <div class="pending-wake-actions">
            <button class="button primary" type="button" data-finalize-wake="${escapeHtml(wake.wake_id)}">发布到正式记忆</button>
            <button class="button" type="button" data-abandon-wake="${escapeHtml(wake.wake_id)}">丢弃</button>
          </div>
          ${result ? `<p class="pending-wake-result${error ? ' error' : ''}">${escapeHtml(result)}</p>` : ''}
        </article>`;
    }).join('');
    root.querySelectorAll('[data-finalize-wake]').forEach((button) => {
      button.addEventListener('click', () => finalizePendingWake(button.dataset.finalizeWake, button));
    });
    root.querySelectorAll('[data-abandon-wake]').forEach((button) => {
      button.addEventListener('click', () => abandonPendingWake(button.dataset.abandonWake, button));
    });
  }

  async function loadPendingWakes(
    epoch = state.selectionEpoch,
    selectedProfileId = state.profile?.id,
  ) {
    if (!selectedProfileId) return;
    const root = $('#pending-wakes-list');
    root.textContent = '正在检查待发布内容…';
    try {
      const data = await api(`/api/profiles/${encodeURIComponent(selectedProfileId)}/pending-wakes`);
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      state.pendingWakes = data.wakes || [];
      renderPendingWakes(state.pendingWakes);
    } catch (error) {
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      root.innerHTML = `<p class="pending-empty">无法读取待发布内容：${escapeHtml(error.message)}</p>`;
    }
  }

  async function finalizePendingWake(wakeId, button) {
    if (!state.profile || !wakeId) return;
    const epoch = state.selectionEpoch;
    const selectedProfileId = state.profile.id;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '正在发布…';
    try {
      const outcome = await api(
        `/api/profiles/${encodeURIComponent(selectedProfileId)}/pending-wakes/${encodeURIComponent(wakeId)}/finalize`,
        { method: 'POST' },
      );
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      state.pendingWakeOutcomes.set(wakeId, outcome);
      const published = ['published', 'already_published'].includes(outcome.status);
      toast(published ? '待发布内容已进入正式记忆。' : pendingOutcomeText(outcome), !published);
      await refreshProfileMemory(epoch, selectedProfileId);
    } catch (error) {
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      state.pendingWakeOutcomes.set(wakeId, { status: 'error', problems: [error.message] });
      renderPendingWakes(state.pendingWakes);
      toast(error.message, true);
    } finally {
      if (button.isConnected) {
        button.disabled = false;
        button.textContent = originalText;
      }
    }
  }

  async function abandonPendingWake(wakeId, button) {
    if (!state.profile || !wakeId) return;
    if (!window.confirm('丢弃该次运行的暂存内容（不再发布到正式记忆），并释放写租约？')) return;
    const epoch = state.selectionEpoch;
    const selectedProfileId = state.profile.id;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '正在丢弃…';
    try {
      const outcome = await api(
        `/api/profiles/${encodeURIComponent(selectedProfileId)}/pending-wakes/${encodeURIComponent(wakeId)}/abandon`,
        { method: 'POST' },
      );
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      const discarded = ['abandoned', 'already_abandoned'].includes(outcome.status);
      if (discarded) {
        state.pendingWakeOutcomes.delete(wakeId);
        state.pendingWakes = (state.pendingWakes || []).filter((wake) => wake.wake_id !== wakeId);
        renderPendingWakes(state.pendingWakes);
        toast('已丢弃该次暂存内容，写租约已释放。');
        await refreshProfileMemory(epoch, selectedProfileId);
      } else {
        state.pendingWakeOutcomes.set(wakeId, { status: outcome.status, problems: [outcome.status] });
        renderPendingWakes(state.pendingWakes);
        toast(`丢弃未完成：${escapeHtml(outcome.status)}`, true);
      }
    } catch (error) {
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      toast(error.message, true);
    } finally {
      if (button.isConnected) {
        button.disabled = false;
        button.textContent = originalText;
      }
    }
  }

  async function searchMemory(event) {
    event.preventDefault();
    if (!state.profile) return;
    const epoch = state.selectionEpoch;
    const selectedProfileId = state.profile.id;
    const query = new FormData(event.currentTarget).get('query') || '';
    try {
      const data = await api(`/api/profiles/${encodeURIComponent(selectedProfileId)}/memory/objects?query=${encodeURIComponent(query)}&limit=50`);
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      renderMemoryObjects(data.items || [], String(query));
      updateObjectSeedOptions(data.items || []);
    } catch (error) {
      if (currentProfileRequest(epoch, selectedProfileId)) toast(error.message, true);
    }
  }

  async function openMemoryObject(id) {
    if (!state.profile) return;
    const epoch = state.selectionEpoch;
    const selectedProfileId = state.profile.id;
    try {
      const detail = await api(`/api/profiles/${encodeURIComponent(selectedProfileId)}/memory/objects/${encodeURIComponent(id)}`);
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      state.memoryObjectId = id;
      const object = detail.object || {};
      $('#memory-detail-title').textContent = object.canonical_name || id;
      const assertions = detail.assertions || [];
      const inquiries = detail.inquiries || [];
      const observations = detail.observations || [];
      $('#memory-detail-content').innerHTML = `
        <p class="memory-object-meta">${escapeHtml(object.kind || 'object')} · ${escapeHtml(id)}</p>
        <section class="memory-detail-section"><h3>当前判断</h3>${assertions.length ? assertions.map((item) => {
          const evidence = Array.isArray(item.evidence) ? item.evidence : [];
          const direction = item.direction === 'subject' ? '当前对象是主语' : '当前对象是宾语';
          return `<article><strong>${escapeHtml(item.predicate || '判断')} ${escapeHtml(item.literal || item.object_name || '')}</strong>
            <span>角色：${escapeHtml(item.epistemic_role || '—')} · 置信度：${escapeHtml(display(item.confidence, '—'))} · ${direction}</span>
            <span>事件时间：${escapeHtml(item.event_time_start || '—')} 至 ${escapeHtml(item.event_time_end || '—')}</span>
            <span>修正关系：supersedes ${escapeHtml(item.supersedes_id || '—')} · superseded_at ${escapeHtml(formatLocalTime(item.superseded_at) || '—')}</span>
            <details><summary>查看依据（${display(evidence.length, 0)} 条${item.evidence_truncated ? '，已截断' : ''}）</summary>${evidence.length ? evidence.map((link) => {
              const obs = link.observation || {};
              return `<p><strong>${escapeHtml(link.role || '—')}</strong> · ${escapeHtml(obs.id || link.observation_id || '—')} · ${escapeHtml(obs.title || '—')}<br>${escapeHtml(obs.source_kind || '—')} · ${escapeHtml(obs.depth || '—')} · 发布 ${escapeHtml(obs.source_published_at || '—')} · 观察 ${escapeHtml(obs.observed_at || '—')}<br>${escapeHtml(obs.source_uri || '—')}<br>材料可靠性：${escapeHtml(obs.material_reliability || '—')}；限制：${escapeHtml((obs.limitations || []).join('；') || '—')}</p>`;
            }).join('') : '<p>没有有效依据链接。</p>'}</details></article>`;
        }).join('') : '<p>暂无当前判断。</p>'}${detail.assertions_truncated ? '<p class="form-help">当前判断列表已截断。</p>' : ''}</section>
        <section class="memory-detail-section"><h3>保留的疑问</h3>${inquiries.length ? inquiries.map((item) => `
          <article><strong>${escapeHtml(item.prompt)}</strong><span>${escapeHtml(item.status)}</span></article>
        `).join('') : '<p>暂无疑问。</p>'}${detail.inquiries_truncated ? '<p class="form-help">保留的疑问列表已截断。</p>' : ''}</section>
        <section class="memory-detail-section"><h3>关联材料</h3>${observations.length ? observations.slice(0, 12).map((item) => `
          <article><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.source_kind)} · ${escapeHtml(item.depth)} · ${escapeHtml(item.material_reliability || 'unknown')}</span><p>${escapeHtml(item.excerpt)}</p></article>
        `).join('') : '<p>暂无可展示的关联材料。</p>'}${observations.length > 12 || detail.observations_truncated ? '<p class="form-help">关联材料为有界列表；当前仅展示前 12 条。</p>' : ''}</section>`;
      openDrawer($('#memory-drawer'));
    } catch (error) {
      if (currentProfileRequest(epoch, selectedProfileId)) toast(error.message, true);
    }
  }

  function useMemoryObject() {
    if (!state.memoryObjectId) return;
    setMode('deep');
    const select = $('#run-form').elements.object_id;
    if (![...select.options].some((option) => option.value === state.memoryObjectId)) {
      select.add(new Option(state.memoryObjectId, state.memoryObjectId));
    }
    select.value = state.memoryObjectId;
    captureRunDraft('deep');
    closeDrawers();
    showWorkspacePanel('start');
    $('#wake-options').hidden = false;
    $('#run-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
    loadPrompt();
    toast('已明确选择为本次 Deep 起点；浏览认知本身不会自动注入。');
  }

  function renderRunInspection(inspection) {
    if (!inspection || !Object.keys(inspection).length) return '';
    const execution = inspection.execution || {};
    const model = inspection.model || {};
    const review = inspection.review || {};
    const writes = inspection.writes || {};
    const durable = inspection.durable || {};
    const issues = inspection.issues || {};
    const writeCounts = {
      objects: Number(durable.objects ?? writes.objects?.length ?? 0),
      assertions: Number(durable.assertions ?? writes.assertions?.length ?? 0),
      inquiries: Number(durable.inquiries ?? writes.inquiries?.length ?? 0),
    };
    const written = writeCounts.objects + writeCounts.assertions + writeCounts.inquiries;
    const tools = inspection.tools || {};
    const byName = tools.by_name || tools;
    const toolRows = Object.entries(byName).map(([name, value]) => {
      const calls = typeof value === 'object' ? value.calls || 0 : value;
      return `<span class="inspection-tool-chip">${escapeHtml(name)} <strong>${escapeHtml(calls)}</strong></span>`;
    }).join('');
    const diagnosticRows = (tools.diagnostics || []).map((item) => {
      const error = item.error || {};
      const label = item.outcome || error.code || (item.ok ? 'success' : 'failed');
      const scope = item.scope && Object.keys(item.scope).length ? JSON.stringify(item.scope) : '';
      const completeness = item.completeness && Object.keys(item.completeness).length
        ? JSON.stringify(item.completeness)
        : '';
      const errorFacts = [
        error.code,
        error.field ? `field=${error.field}` : '',
        error.stage ? `stage=${error.stage}` : '',
        error.attempts !== undefined ? `attempts=${error.attempts}` : '',
      ].filter(Boolean).join(' · ');
      return `<article class="inspection-note"><header><strong>${escapeHtml(item.name || 'tool')} · ${escapeHtml(label)}</strong>
        <small>${escapeHtml(item.call_id || '')}</small></header>
        ${errorFacts ? `<span>错误：${escapeHtml(errorFacts)}</span>` : ''}
        ${scope ? `<span>范围：${escapeHtml(scope)}</span>` : ''}
        ${completeness ? `<span>返回：${escapeHtml(completeness)}</span>` : ''}
        ${error.change_condition ? `<span>变化条件：${escapeHtml(error.change_condition)}</span>` : ''}
        ${(item.limitations || []).length ? `<span>限制：${escapeHtml(item.limitations.join('；'))}</span>` : ''}</article>`;
    }).join('');
    const readableValue = (value) => {
      if (value === null || value === undefined || value === '') return '';
      return typeof value === 'object' ? JSON.stringify(value) : String(value);
    };
    const exactWrites = [
      ...(writes.objects || []).map((item) => ({ type: '对象', text: item.canonical_name || item.id })),
      ...(writes.assertions || []).map((item) => ({
        type: '断言',
        text: item.object_id
          ? [item.subject_name || item.subject_id, item.predicate, item.object_name || item.object_id].filter(Boolean).join(' → ')
          : readableValue(item.literal) || item.predicate,
      })),
      ...(writes.inquiries || []).map((item) => ({ type: '质询', text: item.prompt || item.id })),
    ];
    const visibleWrites = exactWrites.slice(0, 18).map((item) => `
      <article class="inspection-write"><span>${escapeHtml(item.type)}</span><strong>${escapeHtml(item.text || '—')}</strong></article>
    `).join('');
    const writesTruncated = exactWrites.length > 18
      ? '<p class="form-help">具体条目较多，当前展示前 18 条；上方回执数量仍是本轮完整统计。</p>'
      : '';
    const toolCallTotal = tools.total ?? Object.values(byName).reduce(
      (sum, value) => sum + Number(typeof value === 'object' ? value.calls || 0 : value || 0),
      0,
    );
    const rawStopReason = String(execution.stop_reason || '');
    const stopReason = rawStopReason === 'Graph Shell finalize terminal: published'
      ? '正式发布完成'
      : rawStopReason || '—';
    return `
      <details class="run-inspection">
        <summary><span><strong>本轮明细与诊断</strong><small>工具调用、正式写入和停止原因</small></span><b>${escapeHtml(written)} 项写入</b></summary>
        <div class="inspection-grid">
          <span><strong>${escapeHtml(model.successful_calls ?? model.calls ?? 0)}</strong><small>模型调用</small></span>
          <span><strong>${escapeHtml(toolCallTotal)}</strong><small>工具调用</small></span>
          <span><strong>${escapeHtml(written)}</strong><small>正式写入</small></span>
          <span><strong>${escapeHtml(stopReason)}</strong><small>停止原因</small></span>
        </div>
        ${toolRows ? `<section class="inspection-section"><h3>工具调用</h3><div class="inspection-tools">${toolRows}</div></section>` : ''}
        ${diagnosticRows ? `<section class="inspection-section"><h3>工具反馈</h3><div class="inspection-diagnostics">${diagnosticRows}</div>${tools.diagnostics_truncated ? '<p class="form-help">还有更多工具反馈未在此处展开。</p>' : ''}</section>` : ''}
        <section class="inspection-section"><div class="inspection-section-heading"><h3>正式写入</h3><span>对象 ${escapeHtml(writeCounts.objects)} · 断言 ${escapeHtml(writeCounts.assertions)} · 质询 ${escapeHtml(writeCounts.inquiries)}</span></div>
          <div class="inspection-writes">${visibleWrites || '<p class="form-help">本轮没有可精确归属的认知写入。</p>'}</div>${writesTruncated}
        </section>
        ${review.attempts ? `<p class="inspection-footnote">提交审查 ${escapeHtml(review.attempts)} 次</p>` : ''}
        ${(issues.soft || []).length ? `<p class="inspection-note">写入审查提示（不代表事实错误）：${escapeHtml(issues.soft.join('；'))}</p>` : ''}
      </details>`;
  }

  function renderRun(run) {
    const root = $('#run-monitor');
    if (!run) {
      root.className = 'run-monitor empty-state';
      root.innerHTML = '<strong>还没有运行记录</strong><span>启动一次观察后，这里会显示阶段、轮次、Token、成本和实时事件。</span>';
      return;
    }
    const metrics = run.metrics || {};
    const events = run.events || [];
    const outcome = run.outcome || {};
    const result = run.result_summary || {};
    const written = outcome.written || {};
    const tokens = Number(metrics.prompt_tokens || 0) + Number(metrics.completion_tokens || 0);
    const latestContextTokens = Number(metrics.latest_prompt_tokens || 0);
    const latestCachedTokens = Number(metrics.latest_cached_input_tokens || 0);
    const cache = metrics.prompt_tokens
      ? `${Math.round((metrics.cached_input_tokens || 0) / metrics.prompt_tokens * 100)}%`
      : '—';
    const formatCount = (value) => Number.isFinite(Number(value))
      ? Number(value).toLocaleString('zh-CN')
      : display(value);
    const metric = (label, value, hint = '') => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(display(value))}</strong>${hint ? `<small>${escapeHtml(hint)}</small>` : ''}</div>`;
    const statusNames = { queued: '排队中', running: '运行中', succeeded: '已完成', failed: '失败' };
    const phaseNames = { exploration: '探索中', consolidation: '收束中', repair: '修复中', amendment: '修订中' };
    let outcomeMessage = outcome.message || '';
    if (outcome.amendment_failed) outcomeMessage = '原始安全子集已成功写入；后续修订未被采纳，已写入内容保持不变。';
    else if (outcome.durable) outcomeMessage = '本次认知已正式写入该 Agent 的世界记忆；数量来自最终提交回执。';
    else if (run.status === 'failed') outcomeMessage = run.error || '运行未正常完成，请展开明细查看停止原因。';
    else if (activeStatus(run)) outcomeMessage = '正在收集模型调用、工具进度与写入结果。';
    else if (!activeStatus(run)) outcomeMessage = '本次运行正常结束，没有产生持久化变更。';
    const activePhase = activeStatus(run) ? phaseNames[metrics.latest_phase] || metrics.latest_phase : '已结束';
    const outcomeIcons = { success: '✓', warning: '!', error: '×', active: '◎', neutral: '—' };
    const outcomeLevel = outcome.level || 'neutral';
    const emptyEvents = activeStatus(run)
      ? '<div class="empty-state"><strong>等待运行事件</strong><span>模型调用和工具进度会显示在这里。</span></div>'
      : '<div class="empty-state"><strong>没有可恢复的实时事件</strong><span>服务重启后仍保留最终写入、Token 与成本，瞬时进度事件不会跨进程保存。</span></div>';
    const contextHint = latestContextTokens
      ? `最后一次调用输入 · 其中缓存 ${formatCount(latestCachedTokens)}`
      : '最近一次已完成模型调用的输入';
    root.className = 'run-monitor';
    root.innerHTML = `
      <section class="run-result-panel" aria-label="最终写入结果">
        <div class="run-outcome ${escapeHtml(outcome.level || '')}">
          <span class="outcome-icon" aria-hidden="true">${escapeHtml(outcomeIcons[outcomeLevel] || '—')}</span>
          <div class="outcome-copy"><strong>${escapeHtml(outcome.label || statusNames[run.status] || run.status)}</strong><p>${escapeHtml(outcomeMessage)}</p></div>
        </div>
        <dl class="outcome-written"><div><dt>对象</dt><dd>${escapeHtml(formatCount(display(written.objects, 0)))}</dd></div><div><dt>断言</dt><dd>${escapeHtml(formatCount(display(written.assertions, 0)))}</dd></div><div><dt>质询</dt><dd>${escapeHtml(formatCount(display(written.inquiries, 0)))}</dd></div></dl>
      </section>
      <section class="run-metrics-panel" aria-label="运行与 Token 指标">
        <div class="run-metric-group">
          <div class="run-group-heading"><strong>运行概况</strong><span>生命周期与累计成本</span></div>
          <div class="run-cards run-overview-cards">
            ${metric('处理状态', activePhase || '等待中', activeStatus(run) ? '实时阶段' : '运行已结束')}
            ${metric('模型调用', formatCount(metrics.calls || 0), '本次运行累计')}
            ${metric('轮次', metrics.latest_turn ?? result.turn_count, '最后记录轮次')}
            ${metric('成本', metrics.cost_usd !== undefined ? `$${Number(metrics.cost_usd).toFixed(3)}` : result.total_cost_usd !== undefined ? `$${Number(result.total_cost_usd).toFixed(3)}` : '—', '本次运行累计')}
          </div>
        </div>
        <div class="run-metric-group token-metric-group">
          <div class="run-group-heading"><strong>Token 使用</strong><span>当前轮与累计值分开统计</span></div>
          <div class="run-cards run-token-cards">
            ${metric('当前轮上下文', latestContextTokens ? formatCount(latestContextTokens) : '—', contextHint)}
            ${metric('累计 Token', tokens ? formatCount(tokens) : '—', '输入 + 输出累计，非峰值')}
            ${metric('累计输出', metrics.completion_tokens ? formatCount(metrics.completion_tokens) : '—', '全部模型调用输出')}
            ${metric('缓存命中', cache, '缓存输入 / 全部输入')}
          </div>
        </div>
      </section>
      <section class="run-events-section" aria-label="运行事件">
        <div class="run-section-heading"><div><strong>运行事件</strong><span>最近 ${Math.min(events.length, 8)} 条</span></div></div>
        <div class="event-list">
          ${events.length ? events.slice(0, 8).map((event) => `
            <div class="event"><time>${escapeHtml(formatLocalTime(event.at) || '事件')}</time><span>${escapeHtml(event.message || event.kind || '运行事件')}</span></div>
          `).join('') : emptyEvents}
        </div>
      </section>
      ${renderRunInspection(run.inspection)}
    `;
  }

  function setPolling(enabled) {
    if (state.poller) clearInterval(state.poller);
    state.poller = enabled ? setInterval(loadRuns, 2000) : null;
  }

  async function loadRuns(epoch = state.selectionEpoch, selectedProfileId = state.profile?.id) {
    if (!selectedProfileId) return;
    try {
      const previousWasActive = activeStatus(state.activeRun);
      const result = await api('/api/runs');
      if (!currentProfileRequest(epoch, selectedProfileId)) return;
      const runs = Array.isArray(result) ? result : (result.items || result.runs || []);
      const globalActiveRun = runs.find(activeStatus) || null;
      const matches = runs.filter((run) => !run.profile_id || run.profile_id === selectedProfileId);
      let activeRun = matches.find(activeStatus) || matches[0] || null;
      if (runId(activeRun)) {
        activeRun = await api(`/api/runs/${encodeURIComponent(runId(activeRun))}`);
        if (!currentProfileRequest(epoch, selectedProfileId)) return;
      }
      state.globalActiveRun = globalActiveRun;
      state.activeRun = activeRun;
      renderRun(state.activeRun);
      setLaunchAvailability();
      setPolling(activeStatus(state.globalActiveRun));
      if (previousWasActive && !activeStatus(state.activeRun)) {
        await refreshProfileMemory(epoch, selectedProfileId);
      }
    } catch (error) {
      if (currentProfileRequest(epoch, selectedProfileId)) toast(error.message, true);
    }
  }

  async function refreshProfileMemory(
    epoch = state.selectionEpoch,
    selectedProfileId = state.profile?.id,
  ) {
    if (!selectedProfileId) return;
    const updated = await api(`/api/profiles/${encodeURIComponent(selectedProfileId)}`);
    if (!currentProfileRequest(epoch, selectedProfileId)) return;
    state.profile = updated;
    state.profiles = state.profiles.map((profile) => profile.id === updated.id ? updated : profile);
    renderProfiles();
    renderLobby();
    await Promise.all([
      loadMemory(epoch, selectedProfileId),
      loadMemoryGraph(epoch, selectedProfileId),
      loadPendingWakes(epoch, selectedProfileId),
    ]);
  }

  async function startRun(event) {
    event.preventDefault();
    if (!state.profile) return;
    const form = event.currentTarget;
    const data = Object.fromEntries(new FormData(form).entries());
    data.profile_id = state.profile.id;
    data.mode = state.mode;
    captureRunDraft();
    data.thinking = form.elements.thinking.checked;
    if (!data.thinking || !data.reasoning_effort) delete data.reasoning_effort;
    if (!data.perspective) delete data.perspective;
    if (!data.max_cost_usd) delete data.max_cost_usd;
    if (state.mode !== 'deep' || !data.object_id) delete data.object_id;
    if (!toList(data.adapters).length) {
      toast('请至少选择一个信息来源。', true);
      return;
    }
    const button = $('#start-run-button');
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '正在启动…';
    try {
      state.activeRun = await api('/api/runs', { method: 'POST', body: JSON.stringify(data) });
      renderRun(state.activeRun);
      state.globalActiveRun = state.activeRun;
      setLaunchAvailability();
      setPolling(true);
      toast('运行已启动。');
      showWorkspacePanel('history');
      $('#monitor-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) { toast(error.message, true); }
    finally {
      if (!activeStatus(state.globalActiveRun)) {
        button.disabled = false;
        button.textContent = originalText;
      }
    }
  }

  async function loadSystem() {
    try {
      const data = await api('/api/system');
      const provider = data.provider || {};
      $('#connection-status').textContent = '本地服务已连接';
      $('#provider-model').textContent = provider.model || provider.name || '本地模型';
      $('#provider-key-status').textContent = provider.key_configured ? 'API Key 已配置' : 'API Key 未配置';
      $('#writer-note').textContent = data.limits?.single_writer ? '单写入保护已开启' : '单写入保护未开启';
    } catch (_) {
      $('#connection-status').textContent = '本地服务不可用';
      $('#connection-status').parentElement.classList.add('error');
      $('#provider-model').textContent = 'Provider 不可用';
      $('#provider-key-status').textContent = '请检查控制台进程';
    }
  }

  function renderLocalSettings(data) {
    const configured = data?.configured || {};
    const provider = data?.provider || {};
    const status = (id, ready) => {
      const node = $(id);
      node.textContent = ready ? '已连接' : '未配置';
      node.classList.toggle('configured', Boolean(ready));
    };
    status('#deepseek-connection-status', configured.deepseek);
    status('#bilibili-connection-status', configured.bilibili);
    status('#nga-connection-status', configured.nga);
    const form = $('#local-settings-form');
    form.elements.deepseek_model.value = provider.model || '';
    form.elements.deepseek_base_url.value = provider.base_url || '';
    form.elements.deepseek_api_key.value = '';
    form.elements.bilibili_sessdata.value = '';
    form.elements.nga_cookie.value = '';
  }

  async function loadLocalSettings() {
    try {
      renderLocalSettings(await api('/api/local-settings'));
    } catch (error) {
      toast(`无法读取本地连接：${error.message}`, true);
    }
  }

  async function showConnections() {
    openDrawer($('#connections-drawer'));
    await loadLocalSettings();
  }

  async function saveLocalSettings(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const payload = nonEmptySettings(Object.fromEntries(new FormData(form).entries()));
    if (!Object.keys(payload).length) {
      toast('没有需要保存的新连接内容。');
      return;
    }
    try {
      renderLocalSettings(await api('/api/local-settings', {
        method: 'PUT',
        body: JSON.stringify(payload),
      }));
      await loadSystem();
      toast('连接设置已安全写入本机 .env。');
    } catch (error) {
      toast(error.message, true);
    }
  }

  function bindEvents() {
    $('#new-agent-button').addEventListener('click', () => showProfileEditor());
    document.querySelectorAll('[data-new-agent]').forEach((button) => button.addEventListener('click', () => showProfileEditor()));
    $('#empty-new-agent').addEventListener('click', () => showProfileEditor());
    $('#lobby-new-agent').addEventListener('click', () => showProfileEditor());
    $('#brand-home-button').addEventListener('click', showLobby);
    $('#lobby-nav-button').addEventListener('click', showLobby);
    $('#current-agent-nav-button').addEventListener('click', () => state.profile && enterWorkspace(state.profile.id));
    $('#back-to-lobby').addEventListener('click', showLobby);
    $('#connections-button').addEventListener('click', showConnections);
    $('#local-settings-form').addEventListener('submit', saveLocalSettings);
    document.querySelectorAll('[data-workspace-view]').forEach((button) => {
      button.addEventListener('click', () => showWorkspacePanel(button.dataset.workspaceView));
    });
    [$('#header-settings-button'), $('#context-settings-button')].forEach((button) => {
      button.addEventListener('click', () => state.profile && showProfileEditor(state.profile));
    });
    [$('#preview-prompt-button'), $('#context-prompt-button')].forEach((button) => {
      button.addEventListener('click', async () => { await loadPrompt(); openDrawer($('#prompt-drawer')); });
    });
    document.querySelectorAll('[data-close-drawer]').forEach((button) => button.addEventListener('click', closeDrawers));
    $('#drawer-backdrop').addEventListener('click', closeDrawers);
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeDrawers(); });
    $('#profile-form').addEventListener('submit', saveProfile);
    $('#profile-form').querySelectorAll('.expert-settings summary').forEach((summary) => {
      summary.addEventListener('click', (event) => {
        if (summary.closest('.expert-settings').classList.contains('deferred')) {
          event.preventDefault();
          toast(state.cloningProfileId ? '专业配置会从原 Agent 完整复制。' : '先快速创建；随后可在 Agent 设置中完善专业内容。');
        }
      });
    });
    $('#clone-profile-button').addEventListener('click', showCloneEditor);
    $('#run-form').addEventListener('submit', startRun);
    $('#reset-perspective-button').addEventListener('click', resetPerspective);
    $('#refresh-runs').addEventListener('click', () => loadRuns());
    $('#refresh-memory').addEventListener('click', () => Promise.all([loadMemory(), loadMemoryGraph(), loadPendingWakes()]));
    $('#memory-graph-previous').addEventListener('click', () => rotateMemoryGraph(-1));
    $('#memory-graph-next').addEventListener('click', () => rotateMemoryGraph(1));
    $('#memory-graph-auto').addEventListener('click', () => {
      state.memoryGraphAutoRotate = !state.memoryGraphAutoRotate;
      updateMemoryGraphRotationControls(state.memoryGraph);
      scheduleMemoryGraphRotation();
    });
    const graphStage = $('.graph-stage');
    graphStage.addEventListener('mouseenter', () => {
      state.memoryGraphHovered = true;
      clearMemoryGraphRotationTimer();
    });
    graphStage.addEventListener('mouseleave', () => {
      state.memoryGraphHovered = false;
      scheduleMemoryGraphRotation();
    });
    graphStage.addEventListener('focusin', () => {
      state.memoryGraphHovered = true;
      clearMemoryGraphRotationTimer();
    });
    graphStage.addEventListener('focusout', (event) => {
      if (graphStage.contains(event.relatedTarget)) return;
      state.memoryGraphHovered = false;
      scheduleMemoryGraphRotation();
    });
    $('#memory-graph').addEventListener('click', (event) => {
      if (!event.target.closest('.graph-node')) renderGraphDetail(null);
    });
    document.addEventListener('visibilitychange', scheduleMemoryGraphRotation);
    $('#memory-search-form').addEventListener('submit', searchMemory);
    $('#use-memory-object').addEventListener('click', useMemoryObject);
    $('#reload-prompt').addEventListener('click', () => loadPrompt());
    $('#save-operator').addEventListener('click', saveOperator);
    document.querySelectorAll('.mode-card').forEach((card) => card.addEventListener('click', () => setMode(card.dataset.mode)));
    document.querySelectorAll('[data-role="run-adapter"]').forEach((input) => {
      input.addEventListener('change', () => {
        syncAdapterField('run-adapter', $('#run-form').elements.adapters);
        captureRunDraft();
        updateEffectiveDefaults();
      });
    });
    document.querySelectorAll('[data-role="profile-adapter"]').forEach((input) => {
      input.addEventListener('change', () => syncAdapterField('profile-adapter', $('#profile-form').elements.default_adapters));
    });
    $('#run-form').elements.object_id.addEventListener('change', () => loadPrompt());
    ['perspective', 'max_turns', 'max_cost_usd', 'adapters', 'reasoning_effort'].forEach((name) => {
      $('#run-form').elements[name].addEventListener('input', () => {
        captureRunDraft();
        updateEffectiveDefaults();
      });
    });
    $('#run-form').elements.thinking.addEventListener('change', () => {
      syncReasoningControl(
        $('#run-form').elements.thinking,
        $('#run-form').elements.reasoning_effort,
      );
      captureRunDraft();
      updateEffectiveDefaults();
    });
    $('#profile-form').elements.default_thinking.addEventListener('change', () => {
      syncReasoningControl(
        $('#profile-form').elements.default_thinking,
        $('#profile-form').elements.default_reasoning_effort,
      );
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    bindEvents();
    loadSystem();
    loadProfiles().then(showLobby);
  });
})();
