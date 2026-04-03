// Phase 2 stub — patient session vocabulary
//
// HTTP endpoints planned:
//   POST /session/start  { "patient_first": "...", "patient_last": "..." }
//   GET  /session/terms  → active term list as JSON
//   POST /session/end    → reset session additions
//
// Per-connection runtime correction list (hot-reloadable via arc-mutex swap).
// Implement fully in Phase 2 after MVP is verified end-to-end.
