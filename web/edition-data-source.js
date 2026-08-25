const DEFAULT_ENDPOINT = document.documentElement.dataset.editionEndpoint || "./data/edition.json";
const DEFAULT_CATALOG_ENDPOINT = document.documentElement.dataset.editionCatalog || "./data/editions.json";
const DEFAULT_UPCOMING_ENDPOINT = document.documentElement.dataset.upcomingEndpoint || "./data/upcoming.json";
const DATED_BACKFILL_DAYS = 3;
const SITE_TIME_ZONE = "Asia/Shanghai";
const JSON_FETCH_OPTIONS = Object.freeze({
  cache: "no-store",
  headers: { Accept: "application/json" },
});

function pad2(value) {
  return String(value).padStart(2, "0");
}

async function fetchEdition(endpoint) {
  const response = await fetch(endpoint, JSON_FETCH_OPTIONS);
  if (!response.ok) {
    throw new Error(`Edition request failed with status ${response.status}`);
  }
  const edition = await response.json();
  assertEditionShape(edition);
  return edition;
}

export async function loadEditionCatalog(endpoint = DEFAULT_CATALOG_ENDPOINT) {
  try {
    const response = await fetch(endpoint, JSON_FETCH_OPTIONS);
    if (!response.ok) return { latest: "", editions: [] };
    const catalog = await response.json();
    const editions = Array.isArray(catalog.editions)
      ? catalog.editions
          .filter((item) => /^\d{4}-\d{2}-\d{2}$/.test(item?.date || "") && typeof item?.path === "string")
          .sort((a, b) => a.date.localeCompare(b.date))
      : [];
    const latest = editions.some((item) => item.date === catalog.latest)
      ? catalog.latest
      : editions.at(-1)?.date || "";
    return { latest, editions };
  } catch {
    return { latest: "", editions: [] };
  }
}

export async function loadEditionByDate(date, catalog) {
  const entry = catalog.editions.find((item) => item.date === date);
  if (!entry) throw new Error(`Edition is not available for ${date}`);
  return { edition: await fetchEdition(entry.path), date: entry.date };
}

function siteDateKey(date = new Date()) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: SITE_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

export async function loadUpcomingSchedule(endpoint = DEFAULT_UPCOMING_ENDPOINT) {
  try {
    const response = await fetch(endpoint, JSON_FETCH_OPTIONS);
    if (!response.ok) return [];
    const schedule = await response.json();
    const now = new Date();
    const today = siteDateKey(now);
    const limit = siteDateKey(new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000));

    return Array.isArray(schedule.events)
      ? schedule.events
          .filter((event) => /^\d{4}-\d{2}-\d{2}$/.test(event?.date || ""))
          .filter((event) => event.date >= today && event.date <= limit)
          .sort((a, b) => a.date.localeCompare(b.date) || String(a.time || "").localeCompare(String(b.time || ""), "zh-CN"))
      : [];
  } catch {
    return [];
  }
}

export async function loadEditionBundle() {
  const [catalog, upcoming] = await Promise.all([
    loadEditionCatalog(),
    loadUpcomingSchedule(),
  ]);
  const requestedDate = new URLSearchParams(location.search).get("edition");
  const selectedDate = catalog.editions.some((item) => item.date === requestedDate)
    ? requestedDate
    : catalog.latest;

  if (selectedDate) {
    const selected = await loadEditionByDate(selectedDate, catalog);
    return { ...selected, catalog, upcoming };
  }

  const edition = await loadCurrentEdition();
  return { edition, date: inferEditionDate(edition), catalog, upcoming };
}

export async function loadCurrentEdition(endpoint = DEFAULT_ENDPOINT) {
  // 兼容没有 editions.json 的产出：优先探测最近三天的归档，再回退 edition.json。
  for (let offset = 0; offset < DATED_BACKFILL_DAYS; offset++) {
    const date = new Date();
    date.setDate(date.getDate() - offset);
    const dated = `./data/edition-${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}.json`;
    try {
      return await fetchEdition(dated);
    } catch {
      // 404 或格式错误：继续尝试前一天。
    }
  }
  return fetchEdition(endpoint);
}

function inferEditionDate(edition) {
  const idDate = String(edition.editionId || "").match(/\d{4}-\d{2}-\d{2}/)?.[0];
  if (idDate) return idDate;
  return String(edition.publishedAt || "").slice(0, 10);
}

function assertEditionShape(edition) {
  if (!edition || typeof edition !== "object") {
    throw new Error("Edition must be an object");
  }

  for (const key of ["schemaVersion", "editionId", "publishedAt", "domains", "stories"]) {
    if (!(key in edition)) {
      throw new Error(`Edition is missing required field: ${key}`);
    }
  }

  if (!Array.isArray(edition.domains) || !Array.isArray(edition.stories)) {
    throw new Error("Edition domains and stories must be arrays");
  }
}
