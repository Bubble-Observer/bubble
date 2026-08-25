import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const catalogPath = resolve(webRoot, "data", "editions.json");
const outputPath = resolve(webRoot, "data", "upcoming.json");
const catalog = JSON.parse(await readFile(catalogPath, "utf8"));

const normalize = (value = "") => String(value).trim().replace(/\s+/g, " ");
const pad2 = (value) => String(value).padStart(2, "0");

function eventDate(event, editionDate) {
  if (/^\d{4}-\d{2}-\d{2}$/.test(event.date || "")) return event.date;
  const match = normalize(event.dateLabel).match(/(\d{1,2})\s*月\s*(\d{1,2})\s*日/);
  if (!match) return "";
  const sourceYear = Number(editionDate.slice(0, 4));
  const sourceMonth = Number(editionDate.slice(5, 7));
  const month = Number(match[1]);
  const year = sourceMonth >= 10 && month <= 3 ? sourceYear + 1 : sourceYear;
  return `${year}-${pad2(month)}-${pad2(match[2])}`;
}

function allDomains(edition) {
  return edition.domains.flatMap((group) => [group, ...(group.children || [])]);
}

const eventsById = new Map();
for (const entry of [...catalog.editions].sort((a, b) => a.date.localeCompare(b.date))) {
  const editionPath = resolve(webRoot, entry.path.replace(/^\.\//, ""));
  const edition = JSON.parse(await readFile(editionPath, "utf8"));
  for (const domain of allDomains(edition)) {
    for (const event of domain.upcoming || []) {
      const date = eventDate(event, entry.date);
      if (!date || !normalize(event.title)) continue;
      const eventId = normalize(event.eventId || event.id) || [
        domain.id,
        date,
        normalize(event.time),
        normalize(event.title).toLocaleLowerCase("zh-CN"),
      ].join("::");
      const previous = eventsById.get(eventId) || {};
      eventsById.set(eventId, {
        ...previous,
        ...event,
        eventId,
        date,
        sourceDomainId: domain.id,
        sourceDomainLabel: domain.shortName || domain.name || "",
        sourceEditionDate: entry.date,
      });
    }
  }
}

const events = [...eventsById.values()].sort(
  (a, b) => a.date.localeCompare(b.date) || normalize(a.time).localeCompare(normalize(b.time), "zh-CN"),
);

await writeFile(outputPath, `${JSON.stringify({
  schemaVersion: "1.0",
  sourceLatest: catalog.latest,
  events,
}, null, 2)}\n`, "utf8");

console.log(`Upcoming schedule built (${events.length} unique events from ${catalog.editions.length} editions).`);
