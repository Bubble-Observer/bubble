import { readFile } from "node:fs/promises";

const raw = await readFile(new URL("../data/edition.json", import.meta.url), "utf8");
const edition = JSON.parse(raw);

const required = ["schemaVersion", "editionId", "publishedAt", "coverageWindow", "domains", "stories"];
const missing = required.filter((key) => !(key in edition));
if (missing.length) {
  throw new Error(`Edition is missing: ${missing.join(", ")}`);
}

if (edition.schemaVersion !== "1.1") {
  console.warn(`Warning: schemaVersion ${edition.schemaVersion} — output layer now emits 1.1`);
}

const storyIds = new Set();
const assertSourceUrls = (sources, owner) => {
  if (!Array.isArray(sources)) {
    throw new Error(`${owner} must have a sources array`);
  }
  for (const source of sources) {
    if (typeof source?.url !== "string" || !source.url.startsWith("https://")) {
      throw new Error(`${owner} source URL must be HTTPS: ${source?.url || "<missing>"}`);
    }
  }
};

for (const story of edition.stories) {
  if (!story.id || storyIds.has(story.id)) {
    throw new Error(`Story ids must be present and unique: ${story.id || "<missing>"}`);
  }
  storyIds.add(story.id);
  assertSourceUrls(story.sources, `Story ${story.id}`);
  for (const factIndexes of story.factSourceIndexes || []) {
    for (const index of factIndexes) {
      if (!Number.isInteger(index) || index < 0 || index >= story.sources.length) {
        throw new Error(`Story ${story.id} factSourceIndexes index ${index} out of bounds (sources=${story.sources.length})`);
      }
    }
  }
  if (story.eventTime != null && Number.isNaN(Date.parse(story.eventTime))) {
    throw new Error(`Story ${story.id} eventTime is not parseable: ${story.eventTime}`);
  }
  for (const source of story.sources || []) {
    if (source.eventTime != null && Number.isNaN(Date.parse(source.eventTime))) {
      throw new Error(`Story ${story.id} source eventTime is not parseable: ${source.eventTime}`);
    }
  }
}

for (const group of edition.domains) {
  for (const domain of group.children || []) {
    for (const storyId of domain.storyIds || []) {
      if (!storyIds.has(storyId)) {
        throw new Error(`Domain ${domain.id} references unknown story ${storyId}`);
      }
    }
    for (const [index, brief] of (domain.briefs || []).entries()) {
      assertSourceUrls(brief.sources, `Domain ${domain.id} brief ${index}`);
    }
    for (const [index, upcoming] of (domain.upcoming || []).entries()) {
      assertSourceUrls(upcoming.sources, `Domain ${domain.id} upcoming ${index}`);
    }
  }
}

console.log(`Edition ${edition.editionId} is valid (${edition.stories.length} stories, schema ${edition.schemaVersion}).`);
