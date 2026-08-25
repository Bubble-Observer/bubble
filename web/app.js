import { loadEditionBundle, loadEditionByDate } from "./edition-data-source.js?v=global-upcoming-2";
import { decideSceneWheel } from "./scene-mode.js?v=adaptive-scenes-2";

const main = document.querySelector("#content");
const headerDomain = document.querySelector("#header-domain");
const headerDate = document.querySelector("#header-date");
const cultureDialog = document.querySelector("#culture-dialog");
const cultureDialogContent = document.querySelector("#culture-dialog-content");
const evidenceDialog = document.querySelector("#evidence-dialog");
const evidenceDialogContent = document.querySelector("#evidence-dialog-content");
const analysisDialog = document.querySelector("#analysis-dialog");
const analysisDialogContent = document.querySelector("#analysis-dialog-content");
const briefDialog = document.querySelector("#brief-dialog");
const briefDialogContent = document.querySelector("#brief-dialog-content");
const cover = document.querySelector("#cover");
const coverHeadline = document.querySelector("#cover-headline");
const coverDek = document.querySelector("#cover-dek");
const coverIssue = document.querySelector("#cover-issue");
const coverPills = document.querySelector("#cover-pills");
const toc = document.querySelector("#toc");
const tbDomains = document.querySelector("#tb-domains");
const calendarPop = document.querySelector("#calendar-pop");
const calGrid = document.querySelector("#cal-grid");
const calUpcoming = document.querySelector("#cal-upcoming");
const calOpen = document.querySelector("#cal-open");
const calClose = document.querySelector("#cal-close");
const calendarScrim = document.querySelector("#calendar-scrim");
const dateNow = document.querySelector("#date-now");
const toastEl = document.querySelector("#toast");

let edition;
let editionCatalog = { latest: "", editions: [] };
let currentEditionDate = "";
let globalUpcoming = [];
let deckPanels = [];
let deckLocked = false;
let activeDeckIndex = 0;
let lastRenderedRouteKey = "";
let routeSyncToken = 0;

const sceneModeMedia = matchMedia("(min-width: 1100px) and (min-height: 700px)");

function applySceneMode() {
  const enabled = sceneModeMedia.matches;
  document.documentElement.classList.toggle("is-scene-mode", enabled);
  document.body.classList.toggle("is-scene-mode", enabled);
  return enabled;
}

applySceneMode();
sceneModeMedia.addEventListener("change", () => {
  applySceneMode();
});

const escapeHtml = (value = "") =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const pad2 = (value) => String(value).padStart(2, "0");

const safeUrl = (value = "") => (value.startsWith("https://") ? value : "");

const statusLabels = {
  confirmed: "已确认",
  developing: "持续更新",
  analysis: "分析",
  community: "社区讨论",
  scheduled: "已公布",
};

function editionDateObject(value = currentEditionDate) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (match) return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  return new Date(edition?.publishedAt || value);
}

function formatEditionDate(value = currentEditionDate) {
  const date = editionDateObject(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "long",
    day: "numeric",
    weekday: "long",
  }).format(date);
}

function allLeafDomains() {
  return edition.domains.flatMap((group) => group.children || []);
}

function allLiveDomains() {
  return allLeafDomains().filter((domain) => domain.status === "live");
}

function uniqueBy(items, key) {
  const seen = new Set();
  return items.filter((item) => {
    const value = item?.[key];
    if (!value || seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}

function homeStories() {
  const curatedIds = edition.homepage?.storyIds || edition.homepage?.supportingStoryIds;
  if (Array.isArray(curatedIds) && curatedIds.length) {
    const ids = [edition.homepage?.leadStoryId, ...curatedIds].filter(Boolean);
    return uniqueBy(ids.map((id) => edition.stories.find((story) => story.id === id)).filter(Boolean), "id");
  }
  return uniqueBy(allLiveDomains().flatMap((domain) => storiesForDomain(domain)), "id");
}

function homeBriefs() {
  const liveDomains = allLiveDomains();
  return liveDomains.flatMap((domain) =>
    (domain.briefs || []).map((brief, sourceIndex) => ({
      ...brief,
      sourceDomainId: domain.id,
      sourceDomainLabel: liveDomains.length > 1 ? domain.shortName || domain.name : "",
      sourceIndex,
    })),
  );
}

function homeUpcoming() {
  const liveDomains = allLiveDomains();
  return liveDomains.flatMap((domain) =>
    (domain.upcoming || []).map((event) => ({
      ...event,
      sourceDomainId: domain.id,
      sourceDomainLabel: domain.shortName || domain.name,
      sourceEditionDate: currentEditionDate,
    })),
  );
}

function upcomingForDomain(domain) {
  return (domain?.upcoming || []).map((event) => ({
    ...event,
    sourceDomainId: domain.id,
    sourceDomainLabel: "",
    sourceEditionDate: currentEditionDate,
  }));
}

function findDomain(domainId) {
  return edition.domains.find((domain) => domain.id === domainId) ||
    allLeafDomains().find((domain) => domain.id === domainId);
}

function findDomainGroup(domainId) {
  return edition.domains.find((group) =>
    (group.children || []).some((domain) => domain.id === domainId),
  );
}

function storiesForDomain(domain) {
  return (domain.storyIds || [])
    .map((storyId) => edition.stories.find((story) => story.id === storyId))
    .filter(Boolean);
}

function renderStatus(status) {
  return `<span class="status status--${escapeHtml(status)}">${escapeHtml(statusLabels[status] || status)}</span>`;
}

function renderDailySignal(noteCount = 0) {
  const count = Math.min(Math.max(Number(noteCount) || 0, 0), 8);
  const center = { x: 240, y: 140 };
  const radius = { x: 196, y: 91 };
  const nodes = Array.from({ length: count }, (_, index) => {
    const angle = (-110 + (360 / Math.max(count, 1)) * index) * (Math.PI / 180);
    return {
      x: (center.x + radius.x * Math.cos(angle)).toFixed(1),
      y: (center.y + radius.y * Math.sin(angle)).toFixed(1),
      leading: index === 0,
    };
  });

  return `
    <div class="daily-signal" aria-hidden="true">
      <svg viewBox="0 0 480 280" focusable="false">
        <g class="daily-signal__grid">
          <path class="daily-signal__axis" d="M24 140H456" />
          <ellipse class="daily-signal__orbit daily-signal__orbit--outer" cx="240" cy="140" rx="196" ry="91" />
          <ellipse class="daily-signal__orbit daily-signal__orbit--inner" cx="240" cy="140" rx="126" ry="58" />
        </g>
        <g class="daily-signal__links">
          ${nodes.map((node) => `<path d="M${center.x} ${center.y}L${node.x} ${node.y}" />`).join("")}
        </g>
        <ellipse class="daily-signal__sweep" cx="240" cy="140" rx="196" ry="91" pathLength="1" />
        <g class="daily-signal__center">
          <circle class="daily-signal__center-ring" cx="240" cy="140" r="29" />
          <circle class="daily-signal__center-pulse" cx="240" cy="140" r="14" />
          <circle class="daily-signal__center-dot" cx="240" cy="140" r="5" />
        </g>
        ${nodes.map((node) => `
          <g class="daily-signal__node${node.leading ? " daily-signal__node--leading" : ""}" transform="translate(${node.x} ${node.y})">
            <circle class="daily-signal__node-ring" r="8" />
            <circle class="daily-signal__node-dot" r="2.75" />
          </g>`).join("")}
      </svg>
    </div>`;
}

function renderCompactSourceLinks(sources = []) {
  const links = sources
    .map((source, index) => ({ source, index, url: safeUrl(source?.url || "") }))
    .filter((item) => item.url);
  if (!links.length) return "";
  return `<div class="compact-sources" aria-label="内容来源"><span>来源</span>${links
    .map(({ source, index, url }) => `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer" title="${escapeHtml(source.title || source.publisher || "查看来源")}">${escapeHtml(source.publisher || `来源 ${index + 1}`)} ↗</a>`)
    .join("")}</div>`;
}

function renderStoryCard(story, variant = "standard") {
  return `
    <article class="story-card story-card--${variant}" data-story-id="${escapeHtml(story.id)}">
      <a class="story-card__link" href="#/story/${encodeURIComponent(story.id)}" data-route>
        <div class="story-card__topline">
          <span>${escapeHtml(story.eyebrow)}</span>
          <span>${escapeHtml(story.readTime)}</span>
        </div>
        <h3>${escapeHtml(story.title)}</h3>
        <p>${escapeHtml(story.summary)}</p>
        <div class="story-card__footer">
          <span>${escapeHtml(story.updatedLabel)}</span>
          <span class="text-link">阅读全文 <b aria-hidden="true">↗</b></span>
        </div>
      </a>
    </article>`;
}

function renderStoryMetrics(story) {
  const facts = story.facts?.length || 0;
  const sources = story.sources?.length || 0;
  return `
    <div class="story-metrics" aria-label="报道信息摘要">
      <div><strong>${sources}</strong><span>来源</span></div>
      <div><strong>${facts}</strong><span>事实</span></div>
      <div><strong>${escapeHtml(story.readTime)}</strong><span>阅读时间</span></div>
    </div>`;
}

function renderDomainAtlas() {
  const liveGroups = edition.domains.filter((group) => group.status !== "planned");
  return `${liveGroups
    .map((group, index) => {
      const liveChildren = (group.children || []).filter((domain) => domain.status === "live");
      return `
        <a class="domain-node domain-node--live" href="#/domain/${encodeURIComponent(group.id)}" data-route>
          <span class="domain-node__index">0${index + 1}</span>
          <div>
            <span class="domain-node__label">今日已更新</span>
            <strong>${escapeHtml(group.name)}</strong>
          </div>
          <span class="domain-node__state">${liveChildren.length} 个子领域 ↗</span>
        </a>`;
    })
    .join("")}
    <p class="atlas-note">这里只显示已有内容的领域。</p>`;
}

function renderUpcoming(domainOrEvents) {
  const events = Array.isArray(domainOrEvents) ? domainOrEvents : domainOrEvents?.upcoming || [];
  if (!events.length) return "";
  return buildPagerHtml(events, UPCOMING_PER_PAGE, renderUpcomingItem, "upcoming", "日程分页");
}

function renderUpcomingItem(event) {
  return `
    <li class="upcoming-item">
      <div class="upcoming-time">
        <strong>${escapeHtml(event.dateLabel)}</strong>
        <span>${escapeHtml(event.time)}</span>
      </div>
      <div class="upcoming-copy">
        ${event.sourceDomainLabel ? `<span class="content-domain">${escapeHtml(event.sourceDomainLabel)}</span>` : ""}
        <h3>${escapeHtml(event.title)}</h3>
        <p>${escapeHtml(event.note)}</p>
        ${renderCompactSourceLinks(event.sources)}
      </div>
      ${renderStatus(event.status)}
    </li>`;
}

// 短讯预览：取 detail 的第一句（以 。！？ 结尾）作为自包含摘要，最多 60 字；
// 无句读时退化为整段；超长首句截断并补省略号。空 detail 不产生预览（卡片保持静态）。
function briefPreview(detail) {
  const text = detail.trim();
  if (!text) return "";
  const firstSentence = text.match(/^[\s\S]*?[。！？]/)?.[0] ?? text;
  return firstSentence.length > 60 ? `${firstSentence.slice(0, 60)}…` : firstSentence;
}

function renderBriefCard(brief, index) {
  // 无 detail 的短讯不可展开：卡片保持静态，不显示预览与展开提示
  const expandable = Boolean(brief.detail && brief.detail.trim());
  const attrs = expandable
    ? ` data-brief-open="${index}" data-brief-domain="${escapeHtml(brief.sourceDomainId || "")}" data-brief-index="${Number.isInteger(brief.sourceIndex) ? brief.sourceIndex : index}" role="button" tabindex="0" aria-expanded="false" aria-label="展开短讯：${escapeHtml(brief.title)}"`
    : "";
  return `
    <li class="brief-card${expandable ? " is-expandable" : ""}"${attrs}>
      <div class="brief-card__top">
        <time>${escapeHtml(brief.time)}</time>
        ${renderStatus(brief.status)}
      </div>
      ${brief.sourceDomainLabel ? `<span class="content-domain">${escapeHtml(brief.sourceDomainLabel)}</span>` : ""}
      <p class="brief-card__title">${escapeHtml(brief.title)}</p>
      ${expandable ? `
      <p class="brief-card__preview">${escapeHtml(briefPreview(brief.detail))}</p>
      <span class="brief-card__expand" aria-hidden="true">展开 <b>↗</b></span>` : ""}
    </li>`;
}

function renderBriefGrid(briefs) {
  // 每页 = 一整格 2×3 网格（6 条）；不足 6 条的单页同样固定 3 行，与 upcoming 逐行对齐
  return buildPagerHtml(briefs, BRIEFS_PER_PAGE, renderBriefCard, "briefs", "短讯分页", "ul", "brief-grid", ' aria-label="短讯列表"');
}

function renderTimelineItem(item) {
  return `<li><time>${escapeHtml(item.time)}</time><p>${escapeHtml(item.text)}</p></li>`;
}

/* ---------- 内容分页（即将发生 / 时间线 / 分析） ---------- */

const UPCOMING_PER_PAGE = 2;
const BRIEFS_PER_PAGE = 4; // 保证低高度桌面也能完整显示，不依赖容器内滚动
const ANALYSIS_PAGING_MIN = 4; // 分析与判断 ≥ 4 段时分页，每页 2 段
const ANALYSIS_PER_PAGE = 2;
const TIMELINE_PAGING_MIN = 5;
const TIMELINE_PER_PAGE = 4;
const FACTS_PAGING_MIN = 5;
const FACTS_PER_PAGE = 4;

function buildPagerHtml(items, perPage, renderItem, pagerKey, controlsLabel, pageTag = "ul", pageClass = "", pageAttrs = "") {
  const pages = [];
  for (let index = 0; index < items.length; index += perPage) {
    pages.push(items.slice(index, index + perPage));
  }
  // 所有页同时渲染并叠放在同一网格单元：容器高度 = 最高页，切换页不产生位移。
  // renderItem 第二参为全局下标（跨页连续），供 data-* 下标回查原数组（分页后仍精确）
  const pagesHtml = `<div class="pager-pages">${pages
    .map(
      (page, pageIndex) => `
        <${pageTag} class="pager-page${pageClass ? ` ${pageClass}` : ""}${pageIndex === 0 ? " is-active" : ""}"${pageAttrs} data-pager-panel="${pageIndex}">
          ${page.map((item, localIndex) => renderItem(item, pageIndex * perPage + localIndex)).join("")}
        </${pageTag}>`,
    )
    .join("")}</div>`;
  const controlsHtml = pages.length > 1
    ? `<div class="pager-controls" aria-label="${escapeHtml(controlsLabel)}">
        <span><b data-pager-current>01</b> / ${String(pages.length).padStart(2, "0")}</span>
        <div>
          <button type="button" data-pager-prev aria-label="上一页">←</button>
          <button type="button" data-pager-next aria-label="下一页">→</button>
        </div>
      </div>`
    : `<div class="pager-controls pager-controls--single"><span>共 ${items.length} 条</span></div>`;
  return `<div class="pager" data-pager="${escapeHtml(pagerKey)}">${pagesHtml}${controlsHtml}</div>`;
}

/* ---------- 封面 ---------- */

function renderCover() {
  const date = editionDateObject();
  const dateText = new Intl.DateTimeFormat("zh-CN", {
    year: "numeric", month: "long", day: "numeric",
  }).format(date);
  coverIssue.textContent = `${dateText} · 今日更新`;
  coverHeadline.textContent = edition.headline;
  coverDek.textContent = edition.dek;
  coverPills.innerHTML = edition.domains
    .flatMap((group) => group.children || [])
    .filter((domain) => domain.shortName || domain.name)
    .map((domain) => {
      const live = domain.status === "live";
      const name = escapeHtml(domain.shortName || domain.name);
      return live
        ? `<a class="cover-pill cover-pill--live" href="#/domain/${encodeURIComponent(domain.id)}" data-route>${name} · 今日已更新 ↗</a>`
        : `<span class="cover-pill">${name}</span>`;
    })
    .join("");
}

/* ---------- 头部领域切换 ---------- */

function renderDomainTabs() {
  const route = parseRoute();
  const currentStory = route.kind === "story" ? edition.stories.find((story) => story.id === route.id) : null;
  const current = route.kind === "domain" ? route.id : currentStory?.domainId || "";
  tbDomains.innerHTML = allLiveDomains()
    .filter((domain) => domain.shortName || domain.name)
    .map((domain) => {
      const name = escapeHtml(domain.shortName || domain.name);
      const active = domain.id === current ? " active" : "";
      return `<a class="tb-domain${active}" href="#/domain/${encodeURIComponent(domain.id)}" data-route>${name}</a>`;
    })
    .join("");
}

/* ---------- 日历 ---------- */

function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => toastEl.classList.remove("show"), 1800);
}

function renderCalendar() {
  const editionDate = editionDateObject();
  const year = editionDate.getFullYear();
  const monthIndex = editionDate.getMonth();
  const today = editionDate.getDate();
  const available = new Map(
    editionCatalog.editions
      .filter((item) => item.date.startsWith(`${year}-${pad2(monthIndex + 1)}-`))
      .map((item) => [Number(item.date.slice(-2)), item.date]),
  );
  if (!available.size && currentEditionDate) available.set(today, currentEditionDate);
  const firstDow = new Date(year, monthIndex, 1).getDay(); // 0=周日
  const daysInMonth = new Date(year, monthIndex + 1, 0).getDate();
  const dows = ["日", "一", "二", "三", "四", "五", "六"];
  let html = dows.map((d) => `<span class="cal-dow">${d}</span>`).join("");
  for (let i = 0; i < firstDow; i++) html += `<span class="cal-blank"></span>`;
  for (let day = 1; day <= daysInMonth; day++) {
    const editionDateKey = available.get(day);
    const isAvailable = Boolean(editionDateKey);
    const cls = day === today ? "cal-day cal-day--today" : "cal-day";
    html += `<button type="button" class="${cls}" data-edition-date="${editionDateKey || ""}" ${isAvailable ? "" : "disabled"} aria-label="${day} 日${isAvailable ? "，有当日内容" : "，暂无当日内容"}">${day}</button>`;
  }
  calGrid.innerHTML = html;

  const upcoming = globalUpcoming;
  calUpcoming.innerHTML = upcoming.length
    ? upcoming.slice(0, 6).map((e) => `<div class="cal-event"><b>${escapeHtml(e.dateLabel)}</b><span>${e.sourceDomainLabel ? `${escapeHtml(e.sourceDomainLabel)} · ` : ""}${escapeHtml(e.title)}</span></div>`).join("")
    : `<div class="cal-event"><span style="color:var(--ink-faint)">暂无已公布日程</span></div>`;

  calGrid.querySelectorAll(".cal-day").forEach((btn) => {
    btn.addEventListener("click", () => switchEditionDate(btn.dataset.editionDate));
  });
}

function openCalendar() {
  renderCalendar();
  calendarPop.classList.add("show");
  calendarScrim.hidden = false;
  calOpen.setAttribute("aria-expanded", "true");
  calClose.focus();
}
function closeCalendar() {
  calendarPop.classList.remove("show");
  calendarScrim.hidden = true;
  calOpen.setAttribute("aria-expanded", "false");
}

function updateEditionChrome() {
  const date = editionDateObject();
  headerDate.textContent = formatEditionDate();
  dateNow.textContent = `${date.getMonth() + 1} 月 ${date.getDate()} 日`;
  const currentIndex = editionCatalog.editions.findIndex((item) => item.date === currentEditionDate);
  document.querySelectorAll(".date-btn").forEach((btn) => {
    const nextIndex = currentIndex + Number(btn.dataset.dir || 0);
    btn.disabled = currentIndex < 0 || !editionCatalog.editions[nextIndex];
    btn.title = btn.disabled ? "没有相邻日期的内容" : "";
  });
}

async function switchEditionDate(date) {
  if (!date || date === currentEditionDate) {
    closeCalendar();
    return;
  }

  document.querySelectorAll(".date-btn").forEach((btn) => { btn.disabled = true; });
  try {
    const selected = await loadEditionByDate(date, editionCatalog);
    edition = selected.edition;
    currentEditionDate = selected.date;
    const nextUrl = new URL(location.href);
    nextUrl.searchParams.set("edition", currentEditionDate);
    nextUrl.hash = "#/";
    history.pushState({}, "", nextUrl);
    renderCover();
    updateEditionChrome();
    renderRoute({ focus: true, resetScroll: true, force: true });
    showToast(`${formatEditionDate()} · 已切换`);
  } catch (error) {
    console.error(error);
    showToast("暂时无法加载该日内容");
    updateEditionChrome();
  } finally {
    closeCalendar();
  }
}
calOpen.addEventListener("click", (event) => {
  event.stopPropagation();
  if (calendarPop.classList.contains("show")) closeCalendar();
  else openCalendar();
});
calClose.addEventListener("click", closeCalendar);
calendarScrim.addEventListener("click", closeCalendar);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (calendarPop.classList.contains("show")) closeCalendar();
  if (evidenceDialog.open) evidenceDialog.close();
  if (analysisDialog.open) analysisDialog.close();
  if (cultureDialog.open) cultureDialog.close();
  if (briefDialog.open) briefDialog.close();
});

document.querySelectorAll(".date-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const currentIndex = editionCatalog.editions.findIndex((item) => item.date === currentEditionDate);
    const next = editionCatalog.editions[currentIndex + Number(btn.dataset.dir || 0)];
    if (next) switchEditionDate(next.date);
  });
});

/* ---------- 封面滚动（轻微视差） ---------- */

function handleScroll() {
  const y = window.scrollY;
  const isHome = document.body.dataset.route === "home";
  const isSceneMode = document.documentElement.classList.contains("is-scene-mode");
  const beyondCover = !isHome || y >= window.innerHeight * 0.72;
  document.body.classList.toggle("is-beyond-cover", beyondCover);
  toc.hidden = isHome && (!beyondCover || !isSceneMode);
  if (isSceneMode && deckPanels.length) activeDeckIndex = currentDeckIndex();
}
window.addEventListener("scroll", handleScroll, { passive: true });

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    applySceneMode();
    refreshDeckPanels();
    if (document.documentElement.classList.contains("is-scene-mode")) {
      forceDeckPosition(deckPanels[activeDeckIndex]?.offsetTop || 0);
    }
  }, 120);
});

/* ---------- 目录 scrollspy ---------- */

let spy;
function buildToc(items, container = toc) {
  container.innerHTML = `
    <span class="section-kicker">目录</span>
    <ol class="toc-list">
      ${items.map((item) => `<li><button class="toc-item" type="button" data-scene-target="${item.id}" data-toc-target="${item.id}">${escapeHtml(item.label)}</button></li>`).join("")}
    </ol>`;
  const targets = items.map((item) => document.getElementById(item.id)).filter(Boolean);
  if (spy) spy.disconnect();
  if (!targets.length) return;
  spy = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        container.querySelectorAll(".toc-item").forEach((link) => {
          link.classList.toggle("active", link.dataset.tocTarget === entry.target.id);
        });
      }
    });
  }, { rootMargin: "-25% 0px -65% 0px" });
  targets.forEach((target) => spy.observe(target));
}

function refreshDeckPanels() {
  const route = parseRoute();
  if (route.kind === "home") {
    deckPanels = [cover, ...main.querySelectorAll(".home-scene")];
  } else if (route.kind === "story") {
    deckPanels = [
      main.querySelector(".article-hero"),
      ...main.querySelectorAll(".article-section"),
    ].filter(Boolean);
  } else {
    deckPanels = [];
  }
}

function currentDeckIndex() {
  if (!deckPanels.length) return -1;
  return deckPanels.reduce((current, panel, index) => (
    panel.getBoundingClientRect().top <= 2 ? index : current
  ), 0);
}

function forceDeckPosition(top) {
  document.documentElement.classList.add("is-scene-jump");
  window.scrollTo({ top, behavior: "auto" });
  window.requestAnimationFrame(() => window.requestAnimationFrame(() => document.documentElement.classList.remove("is-scene-jump")));
}

function goToDeckPanel(panel) {
  if (!panel) return;
  const panelIndex = deckPanels.indexOf(panel);
  if (panelIndex >= 0) activeDeckIndex = panelIndex;
  const top = window.scrollY + panel.getBoundingClientRect().top;
  const targetTop = Math.max(0, top);
  const isAdjacent = Math.abs(targetTop - window.scrollY) <= window.innerHeight * 1.25;
  if (!isAdjacent) {
    forceDeckPosition(targetTop);
  } else {
    window.scrollTo({ top: targetTop, behavior: "smooth" });
    window.setTimeout(() => {
      if (panel.isConnected) forceDeckPosition(targetTop);
    }, 560);
  }
}

function handleDeckWheel(event) {
  if (!document.documentElement.classList.contains("is-scene-mode") || !deckPanels.length || Math.abs(event.deltaY) < 12 || event.ctrlKey) return;
  if (calendarPop.classList.contains("show") || document.querySelector("dialog[open]")) return;
  if (event.target.closest("dialog, .calendar-pop")) return;
  if (deckLocked) {
    event.preventDefault();
    return;
  }
  const current = currentDeckIndex();
  const panel = deckPanels[current];
  if (!panel) return;
  const panelRect = panel.getBoundingClientRect();
  const decision = decideSceneWheel({
    deltaY: event.deltaY,
    panelTop: panelRect.top,
    panelBottom: panelRect.bottom,
    viewportHeight: window.innerHeight,
    currentIndex: current,
    panelCount: deckPanels.length,
  });
  if (decision.action !== "navigate") return;
  event.preventDefault();
  deckLocked = true;
  goToDeckPanel(deckPanels[decision.nextIndex]);
  window.setTimeout(() => { deckLocked = false; }, 720);
}

window.addEventListener("wheel", handleDeckWheel, { passive: false });

/* ---------- 首页 ---------- */

function renderHome() {
  const liveDomains = allLiveDomains();
  const stories = homeStories();
  const configuredLead = edition.homepage?.leadStoryId
    ? stories.find((story) => story.id === edition.homepage.leadStoryId)
    : null;
  const lead = configuredLead || stories.find((story) => story.level === "core") || stories[0];
  const leadDomain = findDomain(lead?.domainId) || liveDomains[0];
  const supporting = stories.filter((story) => story.id !== lead?.id);
  const supportingPages = Array.from({ length: Math.max(1, Math.ceil(supporting.length / 3)) }, (_, index) => supporting.slice(index * 3, index * 3 + 3));

  const briefs = homeBriefs();
  const upcoming = homeUpcoming();
  const pulseLabel = briefs.length && upcoming.length
    ? "简讯与日程"
    : briefs.length
      ? "简讯"
      : "日程";
  const pulseLayoutClass = briefs.length && upcoming.length
    ? ""
    : briefs.length
      ? " home-lower-grid--briefs-only"
      : " home-lower-grid--upcoming-only";
  const pulseScenes = [];
  if (briefs.length) {
    pulseScenes.push(`
        <section class="brief-stream" aria-labelledby="brief-heading">
          <div class="section-heading section-heading--compact">
            <div>
              <span class="section-kicker">更多消息</span>
              <h2 id="brief-heading">短讯</h2>
            </div>
          </div>
          ${renderBriefGrid(briefs)}
        </section>`);
  }
  if (upcoming.length) {
    pulseScenes.push(`
        <section class="upcoming" aria-labelledby="upcoming-heading">
          <div class="section-heading section-heading--compact">
            <div>
              <span class="section-kicker">未来一周</span>
              <h2 id="upcoming-heading">日程</h2>
            </div>
          </div>
          ${renderUpcoming(upcoming)}
        </section>`);
  }

  headerDomain.textContent = liveDomains.length === 1
    ? liveDomains[0].shortName || liveDomains[0].name
    : liveDomains.length
      ? `${liveDomains.length} 个领域`
      : "今日";

  main.innerHTML = `
    <div class="page-home">
      <section class="home-scene home-scene--signal" id="toc-today" aria-labelledby="today-title">
        <div class="scene-index"><span>01</span><span>今日概览</span></div>
        <div class="scene-copy">
          <span class="section-kicker">${escapeHtml(formatEditionDate())}</span>
          <h1 class="visually-hidden" id="today-title">今日概览</h1>
          ${renderDailySignal(edition.todayNotes.length)}
          <p>${escapeHtml(edition.dek)}</p>
          <button class="scene-jump" type="button" data-scene-target="toc-lead">阅读头条 <b aria-hidden="true">↓</b></button>
        </div>
        <aside class="signal-list" aria-label="今日速览">
          <span class="section-kicker">今日要点</span>
          <ol>${edition.todayNotes.map((note, index) => `<li><span>0${index + 1}</span><p>${escapeHtml(note)}</p></li>`).join("")}</ol>
        </aside>
      </section>

      ${lead ? `<section class="home-scene home-scene--lead" id="toc-lead" aria-label="今日头条">
        <div class="scene-index"><span>02</span><span>头条</span></div>
        <div class="lead-context"><span class="section-kicker">${escapeHtml(leadDomain?.name || "今日")} · 今日重点</span><p>${escapeHtml(edition.dek)}</p>${leadDomain ? `<a class="quiet-link" href="#/domain/${encodeURIComponent(leadDomain.id)}" data-route>查看 ${escapeHtml(leadDomain.name)} ↗</a>` : ""}</div>
        ${renderStoryCard(lead, "lead")}
      </section>` : ""}

      <section class="home-scene home-scene--stories" id="toc-supporting" aria-labelledby="important-heading">
        <div class="scene-index"><span>03</span><span>更多报道</span></div>
        <div class="section-heading">
          <div>
            <span class="section-kicker">今日更新</span>
            <h2 id="important-heading">其他值得关注的事</h2>
          </div>
          <span class="section-count"><b>${String(supporting.length).padStart(2, "0")}</b><small>篇</small></span>
        </div>
        <div class="story-rail" data-story-rail>
          <div class="story-rail__pages">
            ${supportingPages.map((page, index) => `<div class="story-grid story-page${index === 0 ? " is-active" : ""}" data-story-page-panel="${index}" ${index === 0 ? "" : "hidden"}>${page.map((story) => renderStoryCard(story)).join("")}</div>`).join("")}
          </div>
          ${supportingPages.length > 1 ? `<div class="story-rail__controls" aria-label="更多报道分页"><span><b data-story-page-current>01</b> / ${String(supportingPages.length).padStart(2, "0")}</span><div><button type="button" data-story-page="prev" aria-label="上一组报道">←</button><button type="button" data-story-page="next" aria-label="下一组报道">→</button></div></div>` : `<div class="story-rail__controls story-rail__controls--single"><span>本页 ${String(supporting.length).padStart(2, "0")} 篇</span></div>`}
        </div>
      </section>

      ${pulseScenes.length ? `<section class="home-scene home-scene--pulse" id="toc-lower" aria-label="${pulseLabel}">
        <div class="scene-index"><span>04</span><span>${pulseLabel}</span></div>
        <div class="home-lower-grid${pulseLayoutClass}">
        ${pulseScenes.join("")}
        </div>
      </section>` : ""}

      <section class="home-scene home-scene--atlas" id="toc-atlas" aria-labelledby="atlas-heading">
        <div class="scene-index"><span>05</span><span>浏览领域</span></div>
        <div class="section-heading">
          <div>
            <span class="section-kicker">全部领域</span>
            <h2 id="atlas-heading">按领域查看</h2>
          </div>
          <p>查看各领域今天更新的内容。</p>
        </div>
        <div class="domain-nodes">${renderDomainAtlas()}</div>
      </section>
    </div>`;

  buildToc([
    { id: "toc-today", label: "今日概览" },
    { id: "toc-lead", label: "头条" },
    { id: "toc-supporting", label: "更多报道" },
    ...(pulseScenes.length ? [{ id: "toc-lower", label: pulseLabel }] : []),
    { id: "toc-atlas", label: "浏览领域" },
  ]);
}

/* ---------- 领域页 ---------- */

function renderDomain(domainId) {
  const domain = findDomain(domainId);
  if (!domain || domain.status !== "live") {
    renderNotFound();
    return;
  }

  if (domain.kind === "group") {
    renderGroup(domain);
    return;
  }

  const group = findDomainGroup(domain.id);
  const stories = storiesForDomain(domain);
  const upcoming = upcomingForDomain(domain);
  headerDomain.textContent = domain.shortName || domain.name;
  main.innerHTML = `
    <div class="page-domain">
      <nav class="breadcrumbs" aria-label="当前位置">
        <a href="#/" data-route>今日首页</a><span>/</span><span>${escapeHtml(group?.name || "领域")}</span><span>/</span><strong>${escapeHtml(domain.name)}</strong>
      </nav>
      <header class="domain-hero section-rule" id="toc-domain-hero">
        <div>
          <span class="section-kicker">领域</span>
          <h1>${escapeHtml(domain.name)}</h1>
        </div>
        <div class="domain-hero__copy">
          <p>${escapeHtml(domain.summary)}</p>
          <span>${escapeHtml(formatEditionDate())} · ${stories.length} 篇</span>
        </div>
      </header>
      <section class="domain-story-list" id="toc-domain-stories" aria-label="领域热点">
        ${stories.map((story, index) => renderStoryCard(story, index === 0 ? "lead" : "standard")).join("")}
      </section>
      ${upcoming.length ? `<section class="upcoming domain-upcoming section-block" id="toc-domain-upcoming" aria-labelledby="domain-upcoming-heading">
        <div class="section-heading">
          <div><span class="section-kicker">未来一周</span><h2 id="domain-upcoming-heading">日程</h2></div>
        </div>
        ${renderUpcoming(upcoming)}
      </section>` : ""}
    </div>`;
  toc.innerHTML = "";
}

function renderGroup(group) {
  const children = (group.children || []).filter((domain) => domain.status === "live");
  headerDomain.textContent = group.name;
  main.innerHTML = `
    <div class="page-domain">
      <nav class="breadcrumbs" aria-label="当前位置"><a href="#/" data-route>今日首页</a><span>/</span><strong>${escapeHtml(group.name)}</strong></nav>
      <header class="domain-hero section-rule">
        <div><span class="section-kicker">领域</span><h1>${escapeHtml(group.name)}</h1></div>
        <div class="domain-hero__copy"><p>选择一个领域继续阅读。</p><span>${children.length} 个领域</span></div>
      </header>
      <section class="group-list" aria-label="领域列表">
        ${children
          .map((domain, index) => {
            const stories = storiesForDomain(domain);
            const lead = stories[0];
            return `
              <article class="group-card">
                <span class="group-card__number">0${index + 1}</span>
                <div class="group-card__identity">
                  <span class="section-kicker">领域</span>
                  <h2><a href="#/domain/${encodeURIComponent(domain.id)}" data-route>${escapeHtml(domain.name)} ↗</a></h2>
                  <p>${escapeHtml(domain.summary)}</p>
                </div>
                ${lead ? `<a class="group-card__lead" href="#/story/${encodeURIComponent(lead.id)}" data-route><span>今日重点</span><strong>${escapeHtml(lead.title)}</strong></a>` : ""}
              </article>`;
          })
          .join("")}
      </section>
    </div>`;
  toc.innerHTML = "";
}

/* ---------- 故事详情 ---------- */

function sourcesForFact(story, factIndex) {
  const mappedIndexes = story.factSourceIndexes?.[factIndex];
  if (Array.isArray(mappedIndexes)) {
    return mappedIndexes.map((index) => story.sources?.[index]).filter(Boolean);
  }
  const sources = story.sources || [];
  const perFact = Math.max(1, Math.ceil(sources.length / Math.max(1, story.facts.length)));
  return sources.slice(factIndex * perFact, (factIndex + 1) * perFact);
}

function renderFactTrigger(story, fact, index) {
  const sources = sourcesForFact(story, index);
  const publishers = [...new Set(sources.map((source) => source.publisher))];
  const preview = publishers.slice(0, 2).join("、") || "待补充来源";
  return `
    <li class="fact-record">
      <button class="fact-trigger" type="button" data-evidence-fact="${index}" data-story-id="${escapeHtml(story.id)}" aria-label="查看事实 ${index + 1} 的 ${sources.length} 条证据">
        <span class="fact-index">F${String(index + 1).padStart(2, "0")}</span>
        <span class="fact-statement">${escapeHtml(fact)}</span>
        <span class="fact-proof"><b>${sources.length} 条证据</b><small>${escapeHtml(preview)}${publishers.length > 2 ? ` 等 ${publishers.length} 个来源` : ""}</small></span>
        <span class="fact-expand" aria-hidden="true">＋</span>
      </button>
    </li>`;
}

const ANALYSIS_PREVIEW_CHARS = 90;

function renderAnalysisBlock(story, item, index) {
  const foldable = item.text.length > ANALYSIS_PREVIEW_CHARS;
  const previewText = foldable
    ? escapeHtml(item.text.slice(0, ANALYSIS_PREVIEW_CHARS))
    : escapeHtml(item.text);
  // 与事实记录同构：整块可点（键盘可达），长文进抽屉；短文就地读完
  return `
    <div class="analysis-block${foldable ? " is-foldable" : ""}"${foldable ? ` data-analysis-open="${index}" data-story-id="${escapeHtml(story.id)}" role="button" tabindex="0" aria-label="阅读完整分析：${escapeHtml(item.label)}"` : ""}>
      <span>${escapeHtml(item.label)}</span>
      <p class="analysis-preview">${previewText}${foldable ? "…" : ""}</p>
      ${foldable ? `<b class="analysis-expand" aria-hidden="true">↗</b>` : ""}
    </div>`;
}

function renderStory(storyId) {
  const story = edition.stories.find((item) => item.id === storyId);
  if (!story) {
    renderNotFound();
    return;
  }

  const domain = findDomain(story.domainId);
  const group = findDomainGroup(story.domainId);
  headerDomain.textContent = domain?.shortName || domain?.name || "报道";

  const allStories = edition.stories;
  const storyIndex = allStories.findIndex((item) => item.id === story.id);
  const prevStory = allStories[storyIndex - 1] || allStories[allStories.length - 1];
  const nextStory = allStories[(storyIndex + 1) % allStories.length];
  const finalScene = story.cultureNotes?.length ? "culture" : "analysis";
  const detailNav = allStories.length > 1 ? `
    <nav class="detail-nav" aria-label="相邻报道">
      <a href="#/story/${encodeURIComponent(prevStory.id)}" data-route><span>上一篇</span><strong>← ${escapeHtml(prevStory.title)}</strong></a>
      <a href="#/story/${encodeURIComponent(nextStory.id)}" data-route><span>下一篇</span><strong>${escapeHtml(nextStory.title)} →</strong></a>
    </nav>` : "";

  // 时间线：超过 5 条时分页（每页 5 条，保持时序）；短则整列展示
  const timelineItems = story.timeline || [];
  const timelineHtml = timelineItems.length
    ? timelineItems.length >= TIMELINE_PAGING_MIN
      ? buildPagerHtml(timelineItems, TIMELINE_PER_PAGE, renderTimelineItem, "timeline", "时间线分页", "ol", "timeline")
      : `<ol class="timeline">${timelineItems.map(renderTimelineItem).join("")}</ol>`
    : "";

  // 分析与判断：≥ 4 段时分页（每页 2 段），段内长文仍走原分析抽屉
  const analysisItems = story.analysis || [];
  const analysisHtml = analysisItems.length >= ANALYSIS_PAGING_MIN
    ? buildPagerHtml(analysisItems, ANALYSIS_PER_PAGE, (item, index) => renderAnalysisBlock(story, item, index), "analysis", "分析分页", "div", "analysis-stack")
    : `<div class="analysis-stack">${analysisItems.map((item, index) => renderAnalysisBlock(story, item, index)).join("")}</div>`;

  const factItems = story.facts || [];
  const factsHtml = factItems.length >= FACTS_PAGING_MIN
    ? buildPagerHtml(factItems, FACTS_PER_PAGE, (fact, index) => renderFactTrigger(story, fact, index), "facts", "事实分页", "ol", "fact-list fact-ledger")
    : `<ol class="fact-list fact-ledger">${factItems.map((fact, index) => renderFactTrigger(story, fact, index)).join("")}</ol>`;

  main.innerHTML = `
    <article class="story-detail">
      <header class="article-hero" id="story-intro" data-detail-story="${escapeHtml(story.id)}">
        <nav class="breadcrumbs" aria-label="当前位置">
          <a href="#/" data-route>今日首页</a><span>/</span>
          ${group ? `<a href="#/domain/${encodeURIComponent(group.id)}" data-route>${escapeHtml(group.name)}</a><span>/</span>` : ""}
          ${domain ? `<a href="#/domain/${encodeURIComponent(domain.id)}" data-route>${escapeHtml(domain.name)}</a><span>/</span>` : ""}
          <strong>报道</strong>
        </nav>
        <div class="article-hero__meta">
          ${renderStatus(story.status)}
          <span>${escapeHtml(story.updatedLabel)}</span>
          <span>${escapeHtml(story.readTime)}</span>
        </div>
        <h1>${escapeHtml(story.title)}</h1>
        <p class="article-dek">${escapeHtml(story.summary)}</p>
        <div class="why-matters">
          <span>为什么值得关注</span>
          <p>${escapeHtml(story.whyItMatters)}</p>
        </div>
        ${renderStoryMetrics(story)}
      </header>

      <div class="article-layout">
        <div class="article-body">
          <section class="article-section" id="toc-facts" aria-labelledby="facts-heading">
            <div class="article-section__heading"><span>01</span><div><h2 id="facts-heading">事实与来源</h2></div></div>
            <p class="fact-room-intro">点击任一事实可查看对应来源。</p>
            ${factsHtml}
          </section>

          ${timelineHtml ? `
            <section class="article-section" id="toc-timeline" aria-labelledby="timeline-heading">
              <div class="article-section__heading"><span>02</span><h2 id="timeline-heading">事件进展</h2></div>
              ${timelineHtml}
            </section>` : ""}

          <section class="article-section" id="toc-analysis" aria-labelledby="analysis-heading">
            <div class="article-section__heading"><span>03</span><div><h2 id="analysis-heading">分析</h2></div></div>
            ${analysisHtml}
            ${finalScene === "analysis" ? detailNav : ""}
          </section>

          ${story.cultureNotes?.length ? `
            <section class="article-section culture-section" id="toc-culture" aria-labelledby="culture-heading">
              <div class="article-section__heading"><span>04</span><h2 id="culture-heading">圈内语境</h2></div>
              <p>补充文中涉及的术语、梗与社区背景。</p>
              <div class="culture-chips">${story.cultureNotes.map((note) => `<button type="button" data-culture-note="${escapeHtml(note.id)}" data-story-id="${escapeHtml(story.id)}"><span>${escapeHtml(note.term)}</span>${escapeHtml(note.short)}<b aria-hidden="true">＋</b></button>`).join("")}</div>
              ${finalScene === "culture" ? detailNav : ""}
            </section>` : ""}
        </div>
      </div>
    </article>`;

  buildToc([
    { id: "story-intro", label: "导语" },
    { id: "toc-facts", label: "事实与来源" },
    ...(timelineHtml ? [{ id: "toc-timeline", label: "事件进展" }] : []),
    { id: "toc-analysis", label: "分析" },
    ...(story.cultureNotes?.length ? [{ id: "toc-culture", label: "圈内语境" }] : []),
  ]);
}

function renderSource(source, index) {
  const url = safeUrl(source.url);
  return `
    <li>
      <span class="source-number">${String(index + 1).padStart(2, "0")}</span>
      <div>
        <span class="source-type">${escapeHtml(source.type)} · ${escapeHtml(source.published)}</span>
        <h3>${escapeHtml(source.publisher)}</h3>
        <p>${escapeHtml(source.title)}</p>
        <small>对应事实：${escapeHtml(source.supports)}</small>
        ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">查看来源 ↗</a>` : `<span class="source-placeholder">链接待补充</span>`}
      </div>
    </li>`;
}

function renderNotFound() {
  headerDomain.textContent = "未找到";
  toc.innerHTML = "";
  main.innerHTML = `
    <section class="not-found">
      <span class="section-kicker">页面不存在</span>
      <h1>这个页面暂时无法访问。</h1>
      <p>内容可能已经更新，或者这个领域尚未开放。</p>
      <a class="primary-action" href="#/" data-route>返回今日首页</a>
    </section>`;
}

function parseRoute() {
  const value = location.hash.replace(/^#\/?/, "");
  const [kind, id] = value.split("/");
  if (!value) return { kind: "home" };
  return { kind, id: decodeURIComponent(id || "") };
}

function renderRoute({ focus = false, resetScroll = false, force = false } = {}) {
  const route = parseRoute();
  const routeKey = `${currentEditionDate}:${route.kind}:${route.id || ""}`;
  if (!force && routeKey === lastRenderedRouteKey) {
    if (resetScroll) forceDeckPosition(0);
    if (focus) main.focus({ preventScroll: true });
    return;
  }

  if (route.kind === "story") renderStory(route.id);
  else if (route.kind === "domain") renderDomain(route.id);
  else renderHome();

  document.body.dataset.route = route.kind;
  document.documentElement.dataset.route = route.kind;
  toc.classList.toggle("toc--story", route.kind === "story");
  renderDomainTabs();
  initPagers();
  refreshDeckPanels();
  activeDeckIndex = 0;
  lastRenderedRouteKey = routeKey;
  if (resetScroll) forceDeckPosition(0);
  handleScroll();
  if (focus) main.focus({ preventScroll: true });
}

function navigate(href, origin) {
  const update = () => {
    closeCalendar();
    history.pushState({}, "", href);
    renderRoute({ focus: true, resetScroll: true });
  };

  if (document.startViewTransition && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const card = origin?.closest(".story-card");
    if (card) card.classList.add("is-transitioning");
    document.startViewTransition(update);
  } else {
    update();
  }
}

function openCultureNote(storyId, noteId) {
  const story = edition.stories.find((item) => item.id === storyId);
  const note = story?.cultureNotes?.find((item) => item.id === noteId);
  if (!note) return;

  cultureDialogContent.innerHTML = `
    <span class="section-kicker">圈内语境</span>
    <h2 id="culture-title">${escapeHtml(note.term)}</h2>
    <p class="culture-lead">${escapeHtml(note.short)}</p>
    <h3>背景</h3>
    <p>${escapeHtml(note.background)}</p>
    <h3>不同说法</h3>
    <ul>${note.interpretations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  cultureDialog.showModal();
}

function openEvidenceFact(storyId, factIndexValue) {
  const story = edition.stories.find((item) => item.id === storyId);
  const factIndex = Number(factIndexValue);
  const fact = story?.facts?.[factIndex];
  if (!story || !fact) return;
  const sources = sourcesForFact(story, factIndex);
  evidenceDialogContent.innerHTML = `
    <div class="evidence-dialog__scroll">
      <span class="section-kicker">事实 ${String(factIndex + 1).padStart(2, "0")} · 来源</span>
      <h2 id="evidence-title">相关来源</h2>
      <p class="evidence-statement">${escapeHtml(fact)}</p>
      <div class="evidence-dialog__summary"><strong>${sources.length}</strong><span>条来源用于核实这条事实</span></div>
      <ol class="source-list evidence-dialog__sources">${sources.length ? sources.map(renderSource).join("") : `<li class="empty-source">尚未补充这条事实对应的来源。</li>`}</ol>
    </div>`;
  evidenceDialog.showModal();
}

function changeStoryPage(control) {
  const rail = control.closest("[data-story-rail]");
  const pages = [...rail.querySelectorAll("[data-story-page-panel]")];
  if (pages.length < 2) return;
  const current = Math.max(0, pages.findIndex((page) => !page.hidden));
  const direction = control.dataset.storyPage === "next" ? 1 : -1;
  const next = (current + direction + pages.length) % pages.length;
  pages.forEach((page, index) => {
    page.hidden = index !== next;
    page.classList.toggle("is-active", index === next);
  });
  const counter = rail.querySelector("[data-story-page-current]");
  if (counter) counter.textContent = String(next + 1).padStart(2, "0");
}

/* ---------- 用户控制的分页 ---------- */

function showPagerPage(pager, nextIndex) {
  pager.index = (nextIndex + pager.pages.length) % pager.pages.length;
  pager.pages.forEach((page, index) => {
    // 页始终叠放在同一网格单元（is-active 控制可见性），高度天然稳定
    page.classList.toggle("is-active", index === pager.index);
  });
  const counter = pager.root.querySelector("[data-pager-current]");
  if (counter) counter.textContent = String(pager.index + 1).padStart(2, "0");
}

function initPagers(container = document) {
  container.querySelectorAll("[data-pager]").forEach((root) => {
    const pages = [...root.querySelectorAll("[data-pager-panel]")];
    if (pages.length < 2) return;
    const pager = { root, pages, index: 0 };
    root.addEventListener("click", (event) => {
      const prev = event.target.closest("[data-pager-prev]");
      const next = event.target.closest("[data-pager-next]");
      if (prev) showPagerPage(pager, pager.index - 1);
      else if (next) showPagerPage(pager, pager.index + 1);
      else return;
    });
  });
}

document.addEventListener("click", (event) => {
  const sceneControl = event.target.closest("[data-scene-target]");
  if (sceneControl) {
    const target = document.getElementById(sceneControl.dataset.sceneTarget);
    goToDeckPanel(target);
    return;
  }

  const storyPageControl = event.target.closest("[data-story-page]");
  if (storyPageControl) {
    changeStoryPage(storyPageControl);
    return;
  }

  const evidenceControl = event.target.closest("[data-evidence-fact]");
  if (evidenceControl) {
    openEvidenceFact(evidenceControl.dataset.storyId, evidenceControl.dataset.evidenceFact);
    return;
  }

  const briefCard = event.target.closest("[data-brief-open]");
  if (briefCard) {
    openBriefDetail(briefCard.dataset.briefDomain, briefCard.dataset.briefIndex, briefCard);
    return;
  }

  const routeLink = event.target.closest("[data-route]");
  if (routeLink && routeLink instanceof HTMLAnchorElement) {
    event.preventDefault();
    if (routeLink.getAttribute("href") !== location.hash) navigate(routeLink.getAttribute("href"), routeLink);
    return;
  }

  const noteButton = event.target.closest("[data-culture-note]");
  if (noteButton) openCultureNote(noteButton.dataset.storyId, noteButton.dataset.cultureNote);
  return;
});

function openBriefDetail(domainId, briefIndexValue, trigger) {
  const domain = findDomain(domainId) || allLiveDomains()[0];
  const brief = domain?.briefs?.[Number(briefIndexValue)];
  if (!brief) return;
  // 复用分析抽屉的阅读结构：滚动区、渐变收尾、陈述排版
  briefDialogContent.innerHTML = `
    <div class="evidence-dialog__scroll">
      <span class="section-kicker">短讯详情</span>
      <h2 id="brief-dialog-title">${escapeHtml(brief.title)}</h2>
      <div class="brief-dialog__meta"><time>${escapeHtml(brief.time)}</time>${renderStatus(brief.status)}</div>
      <p class="evidence-statement">${escapeHtml(brief.detail || "")}</p>
      ${renderCompactSourceLinks(brief.sources)}
    </div>`;
  trigger?.setAttribute("aria-expanded", "true");
  briefDialog.showModal();
}

function openAnalysis(storyId, indexValue) {
  const story = edition.stories.find((item) => item.id === storyId);
  const item = story?.analysis?.[Number(indexValue)];
  if (!item) return;
  // 复用证据抽屉的阅读结构：滚动区、渐变收尾、陈述排版
  analysisDialogContent.innerHTML = `
    <div class="evidence-dialog__scroll">
      <span class="section-kicker">分析</span>
      <h2 id="analysis-title">${escapeHtml(item.label)}</h2>
      <p class="evidence-statement">${escapeHtml(item.text)}</p>
    </div>`;
  analysisDialog.showModal();
}

document.addEventListener("click", (event) => {
  const analysisOpen = event.target.closest("[data-analysis-open]");
  if (!analysisOpen) return;
  openAnalysis(analysisOpen.dataset.storyId, analysisOpen.dataset.analysisOpen);
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const analysisOpen = event.target.closest?.("[data-analysis-open]");
  if (!analysisOpen) return;
  event.preventDefault();
  openAnalysis(analysisOpen.dataset.storyId, analysisOpen.dataset.analysisOpen);
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const briefCard = event.target.closest?.("[data-brief-open]");
  if (!briefCard) return;
  event.preventDefault();
  openBriefDetail(briefCard.dataset.briefDomain, briefCard.dataset.briefIndex, briefCard);
});

document.querySelectorAll(".dialog-close").forEach((button) => {
  button.addEventListener("click", () => button.closest("dialog")?.close());
});
[cultureDialog, evidenceDialog, analysisDialog, briefDialog].forEach((dialog) => {
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    dialog.close();
  });
});

briefDialog.addEventListener("close", () => {
  document.querySelectorAll("[data-brief-open]").forEach((card) => card.setAttribute("aria-expanded", "false"));
});

async function syncRouteFromLocation() {
  const syncToken = ++routeSyncToken;
  const requestedDate = new URLSearchParams(location.search).get("edition");
  if (requestedDate && requestedDate !== currentEditionDate && editionCatalog.editions.some((item) => item.date === requestedDate)) {
    const selected = await loadEditionByDate(requestedDate, editionCatalog);
    if (syncToken !== routeSyncToken) return;
    edition = selected.edition;
    currentEditionDate = selected.date;
    renderCover();
    updateEditionChrome();
  }
  if (syncToken !== routeSyncToken) return;
  renderRoute({ focus: true, resetScroll: true });
}

window.addEventListener("popstate", syncRouteFromLocation);
window.addEventListener("hashchange", () => {
  if (location.hash.startsWith("#/")) syncRouteFromLocation();
});

try {
  const bundle = await loadEditionBundle();
  edition = bundle.edition;
  editionCatalog = bundle.catalog;
  currentEditionDate = bundle.date;
  globalUpcoming = bundle.upcoming;
  updateEditionChrome();
  renderCover();
  applySceneMode();
  renderRoute();
} catch (error) {
  console.error(error);
  showToast("今日内容未能加载，当前展示静态预览");
}
