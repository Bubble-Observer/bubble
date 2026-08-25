# Security

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Report privately
to the repository maintainers (GitHub private vulnerability reporting, or a
direct message to the maintainer listed on the project profile page). You will
receive an acknowledgement within a few days and a status update after that.

## Design posture

- **Offline by default.** Every automated test, evaluation scenario, and the
  offline demo run with no network and no API key. Real source acquisition
  requires an explicit `--replay-fixture`/`--scripted-model-fixture` pair or
  explicitly configured live adapters.
- **Fail closed.** Identity pre-checks, stale-base guards, readiness gates, and
  the publication transaction all refuse rather than guess.
- **Credentials never logged.** API keys and session tokens must not appear in
  prompts, fixtures, databases, or artifacts; the codebase keeps provider
  configuration out of the world databases.
- **No destructive SQL.** The demo and tests always operate on isolated,
  throwaway database copies, never on configured real databases.
