import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const catalogPath = resolve(webRoot, "data", "editions.json");
const fallbackPath = resolve(webRoot, "data", "edition.json");
const upcomingPath = resolve(webRoot, "data", "upcoming.json");
const catalog = JSON.parse(await readFile(catalogPath, "utf8"));

if (!Array.isArray(catalog.editions) || !catalog.editions.length) {
  throw new Error("Edition catalog must contain at least one edition");
}

const seenDates = new Set();
for (const entry of catalog.editions) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(entry.date || "")) {
    throw new Error(`Invalid catalog date: ${entry.date || "<missing>"}`);
  }
  if (seenDates.has(entry.date)) throw new Error(`Duplicate catalog date: ${entry.date}`);
  seenDates.add(entry.date);

  const editionPath = resolve(webRoot, entry.path.replace(/^\.\//, ""));
  const edition = JSON.parse(await readFile(editionPath, "utf8"));
  const idDate = String(edition.editionId || "").match(/\d{4}-\d{2}-\d{2}/)?.[0];
  const publishedAt = Date.parse(String(edition.publishedAt || ""));
  const editionDayStart = Date.parse(`${entry.date}T00:00:00+08:00`);
  if (idDate !== entry.date || Number.isNaN(publishedAt) || publishedAt < editionDayStart) {
    throw new Error(
      `${entry.path} date mismatch: catalog=${entry.date}, editionId=${idDate}, publishedAt=${edition.publishedAt}`,
    );
  }
  if (!Array.isArray(edition.domains) || !Array.isArray(edition.stories)) {
    throw new Error(`${entry.path} must contain domains and stories arrays`);
  }
}

if (!seenDates.has(catalog.latest)) {
  throw new Error(`Catalog latest date is not listed: ${catalog.latest}`);
}

const latestEntry = catalog.editions.find((entry) => entry.date === catalog.latest);
const latestEditionPath = resolve(webRoot, latestEntry.path.replace(/^\.\//, ""));
const [fallback, latestEdition] = await Promise.all([
  readFile(fallbackPath, "utf8").then(JSON.parse),
  readFile(latestEditionPath, "utf8").then(JSON.parse),
]);
for (const field of ["editionId", "publishedAt", "schemaVersion"]) {
  if (fallback[field] !== latestEdition[field]) {
    throw new Error(
      `Fallback ${field} mismatch: fallback=${JSON.stringify(fallback[field])}, latest=${JSON.stringify(latestEdition[field])}`,
    );
  }
}

const upcoming = JSON.parse(await readFile(upcomingPath, "utf8"));
if (upcoming.sourceLatest !== catalog.latest || !Array.isArray(upcoming.events)) {
  throw new Error("Upcoming schedule is missing or was not built from the latest Edition catalog");
}
const seenEventIds = new Set();
for (const event of upcoming.events) {
  if (!event.eventId || seenEventIds.has(event.eventId)) {
    throw new Error(`Invalid or duplicate upcoming event id: ${event.eventId || "<missing>"}`);
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(event.date || "") || !event.sourceEditionDate) {
    throw new Error(`Upcoming event ${event.eventId} has invalid date metadata`);
  }
  seenEventIds.add(event.eventId);
}

console.log(`Edition catalog is valid (${catalog.editions.length} editions, ${upcoming.events.length} upcoming events, latest ${catalog.latest}).`);
