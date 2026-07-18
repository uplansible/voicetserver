// ==UserScript==
// @name         SCHMIDIspeech
// @namespace    https://github.com/local/schmidispeech
// @version      0.1.15
// @description  Local GPU dictation — German medical (Voxtral/voicetserver + Qwen3/schmidiscribe)
// @match        *://*/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_notification
// ==/UserScript==

(function () {
    "use strict";

    // ---- Configuration ----
    const DEFAULT_HOTKEY = "ctrl+shift+d";

    // Two switchable server backends sharing this one frontend. Both speak the
    // same protocol (raw f32 PCM in, partial/final JSON out, "stop" text frame,
    // identical HTTP API); backend-specific settings are shown/hidden based on
    // what GET /config reports.
    const BACKENDS = {
        voxtral: { label: "Voxtral", defaultUrl: "ws://127.0.0.1:8765/asr" },  // voicetserver
        qwen:    { label: "Qwen3",   defaultUrl: "ws://127.0.0.1:8767/asr" },  // schmidiscribe
    };
    function activeBackend() { return GM_getValue("active_backend", "voxtral"); }

    // Server URL + API key are stored per backend. Pre-profile installs stored a
    // single server_url/api_key — those act as the Voxtral profile's fallback.
    // The per-backend accessors also serve the Diktate tab, which can replay a
    // stored dictation through EITHER backend regardless of which one is active.
    function backendWsUrl(b) {
        const legacy = b === "voxtral"
            ? GM_getValue("server_url", BACKENDS.voxtral.defaultUrl)
            : BACKENDS[b].defaultUrl;
        return GM_getValue("server_url_" + b, legacy);
    }
    function backendApiKey(b) {
        const legacy = b === "voxtral" ? GM_getValue("api_key", "") : "";
        return GM_getValue("api_key_" + b, legacy);
    }
    function getServerUrl() { return backendWsUrl(activeBackend()); }
    function setServerUrl(url) { GM_setValue("server_url_" + activeBackend(), url); }
    function getHotkey() { return GM_getValue("hotkey", DEFAULT_HOTKEY); }
    function setHotkey(k) { GM_setValue("hotkey", k.toLowerCase().trim()); }
    function isCommitMode() { return GM_getValue("commit_mode", false); }
    function apiKey() { return backendApiKey(activeBackend()); }
    function setApiKey(k) { GM_setValue("api_key_" + activeBackend(), k.trim()); }
    function getHotwords() { return GM_getValue("hotwords", ""); }
    function setHotwords(v) { GM_setValue("hotwords", v); }

    // Hotwords stored as free text (one per line / comma separated); normalise to a
    // single comma-separated list for the ?hotwords= query parameter.
    function hotwordsForUrl() {
        return getHotwords()
            .split(/[\n,;]+/)
            .map((s) => s.trim())
            .filter(Boolean)
            .join(",");
    }

    // Patient family name, if the host page exposes it (upl.smr.app); sent as
    // ?patient= for schmidiscribe's prompt biasing. Absent element → empty string.
    function getPatientName() {
        const el = document.querySelector("#schmidi-pat-info-patientendaten");
        return (el && el.dataset && el.dataset.nachname) ? el.dataset.nachname.trim() : "";
    }

    // Wrapper around fetch() that injects the X-Api-Key header on every request.
    // All server HTTP calls go through this so the key is always sent.
    function authFetch(url, init = {}) {
        const key = apiKey();
        if (key) {
            init.headers = { ...(init.headers || {}), "X-Api-Key": key };
        }
        return fetch(url, init);
    }

    // ---- State ----
    let ws = null;
    let audioContext = null;
    let scriptProcessor = null;
    let micStream = null;
    let recording = false;
    let reconnectAttempts = 0;
    let reconnectTimer = null;
    let pendingOnReady = null;     // runs on whichever onopen fires first (survives retries)
    let awaitingFinalFlush = false; // "stop" sent — waiting for server final + close
    let stopFinalizeTimer = null;   // 2s fallback if the server never closes
    let lastCfg = {};               // last GET /config response — drives per-backend UI
    const MAX_RECONNECT = 3;
    const SAMPLE_RATE = 16000;

    let targetEl = null;
    let accumulatedText = "";      // live mode: displayed in overlay (finals seen so far)
    let currentPartial = "";       // current unfinalized partial (overlay only)
    let pendingText = "";          // commit mode: accumulated text not yet injected
    let originalTranscribed = "";  // commit mode: model output snapshot for edit-log diff
    let overlayUserEdited = false; // commit mode: user has started editing the overlay div
    let dictationPcmBuffers = [];  // 16 kHz PCM chunks of the running dictation
    let lastDictation = null;      // { pcm, text, saved } — last finished dictation
                                   // (kept so it can be saved as a training-pair candidate)

    // ---- Styles ----
    const style = document.createElement("style");
    style.textContent = `
        @keyframes schmidipulse {
            0%,100% { box-shadow: 0 0 0 0 rgba(192,57,43,0.5); }
            50%      { box-shadow: 0 0 0 10px rgba(192,57,43,0); }
        }
        #schmidi-overlay-partial { color: #aaa; }
        #schmidi-overlay-text {
            white-space: pre-wrap; word-break: break-word;
            outline: none; min-height: 1.5em; caret-color: #fff;
        }
        #schmidi-word-menu {
            position: fixed; background: #1e1e1e; border: 1px solid #555;
            border-radius: 6px; padding: 4px 0; z-index: 2147483647;
            font-size: 13px; color: #ddd; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            min-width: 180px;
        }
        #schmidi-word-menu .sp-menu-item {
            padding: 7px 14px; cursor: pointer; white-space: nowrap;
        }
        #schmidi-word-menu .sp-menu-item:hover { background: #2a2a2a; }
    `;
    document.head.appendChild(style);

    // ---- Mic button ----
    const btn = document.createElement("div");
    btn.id = "schmidispeech-btn";
    Object.assign(btn.style, {
        position: "fixed", bottom: "24px", right: "24px",
        width: "56px", height: "56px", borderRadius: "50%",
        background: "#555", cursor: "pointer", zIndex: "2147483647",
        display: "flex", alignItems: "center", justifyContent: "center",
        boxShadow: "0 2px 8px rgba(0,0,0,0.4)", userSelect: "none",
        transition: "background 0.2s", fontSize: "24px", color: "#fff",
    });
    btn.textContent = "🎤";
    function updateBtnTitle() {
        btn.title = `SCHMIDIspeech [${BACKENDS[activeBackend()].label}] — ` +
            `${getHotkey()} zum Diktieren, Rechtsklick konfigurieren`;
    }
    updateBtnTitle();
    document.body.appendChild(btn);

    // ---- Transcript overlay ----
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", bottom: "90px", right: "24px",
        maxWidth: "360px", minWidth: "180px",
        background: "rgba(20,20,20,0.93)", color: "#eee",
        padding: "10px 14px", borderRadius: "10px",
        fontSize: "14px", lineHeight: "1.5", zIndex: "2147483647",
        display: "none", wordBreak: "break-word", pointerEvents: "none",
        boxShadow: "0 2px 12px rgba(0,0,0,0.5)",
    });

    // Live mode — readonly text display
    const overlayLive = document.createElement("div");
    overlay.appendChild(overlayLive);

    // Commit mode — editable div + partial span + toolbar
    const overlayCommit = document.createElement("div");
    overlayCommit.style.display = "none";

    const overlayTextDiv = document.createElement("div");
    overlayTextDiv.id = "schmidi-overlay-text";
    overlayTextDiv.contentEditable = "true";
    overlayTextDiv.spellcheck = false;
    overlayTextDiv.addEventListener("input", () => { overlayUserEdited = true; });

    const overlayCommitPartial = document.createElement("span");
    overlayCommitPartial.style.cssText = "color:#aaa;pointer-events:none;user-select:none;";

    const overlayToolbar = document.createElement("div");
    overlayToolbar.style.cssText =
        "display:flex;justify-content:flex-end;gap:8px;margin-top:8px;" +
        "border-top:1px solid #333;padding-top:6px;";

    const commitBtn = document.createElement("button");
    commitBtn.textContent = "↵";
    commitBtn.title = "Einfügen";
    commitBtn.style.cssText =
        "background:#2980b9;color:#fff;border:none;border-radius:6px;" +
        "padding:4px 14px;cursor:pointer;font-size:16px;line-height:1;";

    const cancelOverlayBtn = document.createElement("button");
    cancelOverlayBtn.textContent = "✕";
    cancelOverlayBtn.title = "Verwerfen";
    cancelOverlayBtn.style.cssText =
        "background:#333;color:#aaa;border:1px solid #555;border-radius:6px;" +
        "padding:4px 12px;cursor:pointer;font-size:14px;line-height:1;";

    // Save the dictation (audio + current overlay text) as a training-pair
    // candidate for later review in the Diktate tab.
    const trainSaveBtn = document.createElement("button");
    trainSaveBtn.textContent = "💾";
    trainSaveBtn.title = "Als Trainings-Diktat speichern (Review im Diktate-Tab)";
    trainSaveBtn.style.cssText =
        "background:#333;color:#aaa;border:1px solid #555;border-radius:6px;" +
        "padding:4px 12px;cursor:pointer;font-size:14px;line-height:1;";

    overlayToolbar.append(trainSaveBtn, commitBtn, cancelOverlayBtn);
    overlayCommit.append(overlayTextDiv, overlayCommitPartial, overlayToolbar);
    overlay.append(overlayLive, overlayCommit);
    document.body.appendChild(overlay);

    // ---- Right-click word correction on overlay text ----
    overlayTextDiv.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        e.stopPropagation();
        let word = "";
        const sel = window.getSelection();
        if (sel && sel.toString().trim()) {
            word = sel.toString().trim();
        } else {
            let range;
            if (document.caretRangeFromPoint) {
                range = document.caretRangeFromPoint(e.clientX, e.clientY);
            } else if (document.caretPositionFromPoint) {
                const pos = document.caretPositionFromPoint(e.clientX, e.clientY);
                if (pos) { range = document.createRange(); range.setStart(pos.offsetNode, pos.offset); }
            }
            if (range && range.startContainer && range.startContainer.nodeType === Node.TEXT_NODE) {
                const text = range.startContainer.textContent || "";
                const off  = range.startOffset;
                const pre  = (text.slice(0, off).match(/[\wäöüÄÖÜß\-]+$/) || [""])[0];
                const post = (text.slice(off).match(/^[\wäöüÄÖÜß\-]+/) || [""])[0];
                word = pre + post;
            }
        }
        word = word.replace(/[^\wäöüÄÖÜß\-]/g, "").trim();
        if (!word) return;
        showWordMenu(e.clientX, e.clientY, word);
    });

    // ---- Word correction context menu ----
    let wordMenuEl = null;

    function showWordMenu(x, y, word) {
        closeWordMenu();
        wordMenuEl = document.createElement("div");
        wordMenuEl.id = "schmidi-word-menu";
        const item = document.createElement("div");
        item.className = "sp-menu-item";
        item.textContent = `Korrigieren: "${word}"`;
        item.addEventListener("click", () => {
            closeWordMenu();
            const repl = window.prompt(`Ersatzwort für "${word}":`, "");
            if (repl && repl.trim()) addWordPair(word, repl.trim());
        });
        wordMenuEl.appendChild(item);
        document.body.appendChild(wordMenuEl);
        wordMenuEl.style.left = Math.min(x, window.innerWidth  - 210) + "px";
        wordMenuEl.style.top  = Math.min(y, window.innerHeight - 60)  + "px";
        setTimeout(() => document.addEventListener("click", closeWordMenu, { once: true }), 0);
    }

    function closeWordMenu() {
        if (wordMenuEl) { wordMenuEl.remove(); wordMenuEl = null; }
    }

    async function addWordPair(wrong, correct) {
        try {
            await authFetch(`${getHttpBase()}/words`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ add: [`${wrong}=${correct}`], remove: [] }),
            });
            const current = overlayTextDiv.textContent;
            if (current.includes(wrong)) {
                const updated = current.split(wrong).join(correct);
                overlayTextDiv.textContent = updated;
                pendingText = updated;
            }
            const woerterPane = configPanel.querySelector("#schmidi-pane-woerter");
            if (woerterPane && woerterPane.style.display !== "none") loadWords();
            showToast(`Korrektur gespeichert: ${wrong} → ${correct}`);
        } catch (e) {
            showToast("Fehler beim Speichern: " + e.message);
        }
    }

    // ---- Overlay commit / cancel ----

    function clearOverlay() {
        pendingText = "";
        originalTranscribed = "";
        overlayUserEdited = false;
        overlayTextDiv.textContent = "";
        overlayCommitPartial.textContent = "";
        accumulatedText = "";
        currentPartial = "";
        overlay.style.pointerEvents = "none";
        updateOverlay();
    }

    function cancelOverlayWithConfirm() {
        const hasText = pendingText || overlayTextDiv.textContent.trim();
        if (!hasText || window.confirm("Diktat verwerfen? Der transkribierte Text geht verloren.")) {
            clearOverlay();
        }
    }

    commitBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const text = overlayTextDiv.textContent.trimEnd();
        if (text) {
            injectAtCursor(text);
            if (originalTranscribed && text !== originalTranscribed) {
                authFetch(`${getHttpBase()}/log/edit`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        original: originalTranscribed,
                        edited: text,
                        timestamp: new Date().toISOString(),
                    }),
                }).catch(() => {});
            }
        }
        clearOverlay();
    });

    cancelOverlayBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        cancelOverlayWithConfirm();
    });

    trainSaveBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        saveDictationAsReview(overlayTextDiv.textContent.trim());
    });

    // ---- updateOverlay ----
    function updateOverlay() {
        const hasContent = recording || currentPartial || accumulatedText || pendingText ||
            (isCommitMode() && overlayTextDiv.textContent.trim());

        if (!hasContent) {
            overlay.style.display = "none";
            overlay.style.pointerEvents = "none";
            overlayLive.style.display = "none";
            overlayCommit.style.display = "none";
            return;
        }

        overlay.style.display = "block";

        if (isCommitMode()) {
            overlayLive.style.display = "none";
            overlayCommit.style.display = "block";
            overlay.style.pointerEvents = "auto";
            if (!overlayUserEdited) {
                overlayTextDiv.textContent = pendingText;
            }
            overlayCommitPartial.textContent = currentPartial;
        } else {
            overlayLive.style.display = "block";
            overlayCommit.style.display = "none";
            overlay.style.pointerEvents = "none";
            overlayLive.innerHTML =
                escapeHtml(accumulatedText) +
                '<span id="schmidi-overlay-partial">' +
                escapeHtml(currentPartial) +
                "</span>";
        }
    }

    function escapeHtml(s) {
        return s
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    // ---- Config panel ----
    const configPanel = document.createElement("div");
    Object.assign(configPanel.style, {
        position: "fixed", bottom: "90px", right: "24px",
        background: "#1e1e1e", border: "1px solid #444", color: "#ddd",
        padding: "12px", borderRadius: "10px", zIndex: "2147483647",
        display: "none", flexDirection: "column", gap: "8px",
        minWidth: "300px", boxShadow: "0 4px 16px rgba(0,0,0,0.5)", fontSize: "13px",
    });

    const INPUT_STYLE =
        "background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:6px;" +
        "padding:6px 8px;font-size:13px;width:100%;box-sizing:border-box;outline:none;";
    const LABEL_STYLE = "color:#aaa;font-size:11px;";
    const BTN_CANCEL =
        "background:#333;color:#aaa;border:1px solid #555;border-radius:6px;" +
        "padding:4px 12px;cursor:pointer;font-size:12px;";
    const BTN_PRIMARY =
        "background:#2980b9;color:#fff;border:none;border-radius:6px;" +
        "padding:4px 12px;cursor:pointer;font-size:12px;";

    configPanel.innerHTML = `
        <div style="font-weight:bold;margin-bottom:2px;">SCHMIDIspeech <span id="schmidi-server-version" style="color:#888;font-weight:normal;font-size:11px;"></span></div>
        <div style="display:flex;gap:4px;margin-bottom:4px;align-items:center;">
            <span style="${LABEL_STYLE}">Server:</span>
            <button id="schmidi-backend-voxtral" data-backend="voxtral" style="${BTN_CANCEL};flex:1;">Voxtral</button>
            <button id="schmidi-backend-qwen"    data-backend="qwen"    style="${BTN_CANCEL};flex:1;">Qwen3</button>
        </div>
        <div style="display:flex;gap:0;border-bottom:1px solid #444;margin-bottom:4px;flex-wrap:wrap;">
            <button id="schmidi-tab-woerter"       style="background:none;border:none;border-bottom:2px solid transparent;color:#888;padding:4px 8px;cursor:pointer;font-size:12px;">Eigene Wörter</button>
            <button id="schmidi-tab-hotwords"      style="background:none;border:none;border-bottom:2px solid transparent;color:#888;padding:4px 8px;cursor:pointer;font-size:12px;">Hotwords</button>
            <button id="schmidi-tab-aufnehmen"     style="background:none;border:none;border-bottom:2px solid transparent;color:#888;padding:4px 8px;cursor:pointer;font-size:12px;">Aufnehmen</button>
            <button id="schmidi-tab-zweiterdurchgang" style="background:none;border:none;border-bottom:2px solid transparent;color:#888;padding:4px 8px;cursor:pointer;font-size:12px;">2. Durchgang</button>
            <button id="schmidi-tab-training"         style="background:none;border:none;border-bottom:2px solid transparent;color:#888;padding:4px 8px;cursor:pointer;font-size:12px;">Training</button>
            <button id="schmidi-tab-diktate"          style="background:none;border:none;border-bottom:2px solid transparent;color:#888;padding:4px 8px;cursor:pointer;font-size:12px;">Diktate</button>
            <button id="schmidi-tab-einstellungen" style="background:none;border:none;border-bottom:2px solid transparent;color:#888;padding:4px 8px;cursor:pointer;font-size:12px;">Einstellungen</button>
        </div>

        <!-- Hotwords tab (client-side list, sent per session via ?hotwords=) -->
        <div id="schmidi-pane-hotwords" style="display:none;flex-direction:column;gap:6px;">
            <div style="${LABEL_STYLE}">Hotwords (ein Begriff pro Zeile; Prompt-Biasing nur bei Qwen3/schmidiscribe)</div>
            <textarea id="schmidi-hotwords" rows="8" style="${INPUT_STYLE}font-family:monospace;resize:vertical;"></textarea>
            <div id="schmidi-hotwords-status" style="color:#aaa;font-size:11px;min-height:14px;"></div>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="schmidi-cancel-hotwords" style="${BTN_CANCEL}">Schließen</button>
                <button id="schmidi-save-hotwords"   style="${BTN_PRIMARY}">Speichern</button>
            </div>
        </div>

        <!-- 2. Durchgang tab (record already-recorded sentences a second time) -->
        <div id="schmidi-pane-zweiterdurchgang" style="display:none;flex-direction:column;gap:6px;max-width:340px;">
            <div style="display:flex;gap:6px;align-items:center;">
                <button id="sp2-prev" style="${BTN_CANCEL}">◀</button>
                <span id="sp2-index" style="${LABEL_STYLE}">0/0</span>
                <button id="sp2-next" style="${BTN_CANCEL}">▶</button>
            </div>
            <div id="sp2-sentence" style="background:#2a2a2a;border:1px solid #555;border-radius:6px;padding:8px;font-size:13px;line-height:1.4;min-height:40px;color:#eee;">Lade...</div>
            <div style="display:flex;gap:6px;">
                <button id="sp2-record"   style="${BTN_PRIMARY};flex:1;">⏺ Aufnehmen</button>
                <button id="sp2-vorhören" style="${BTN_CANCEL};flex:1;" disabled>▶ Vorhören</button>
            </div>
            <button id="sp2-save" style="${BTN_CANCEL};width:100%;" disabled>💾 Speichern</button>
            <div id="sp2-status" style="${LABEL_STYLE};min-height:14px;"></div>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="sp2-close" style="${BTN_CANCEL}">Schließen</button>
            </div>
        </div>

        <!-- Eigene Wörter tab -->
        <div id="schmidi-pane-woerter" style="display:none;flex-direction:column;gap:6px;">
            <div style="${LABEL_STYLE}">Eigene Wörter (ein Wort pro Zeile; falsch=richtig für Ersetzungen)</div>
            <textarea id="schmidi-words" rows="8" style="${INPUT_STYLE}font-family:monospace;resize:vertical;"></textarea>
            <button id="schmidi-words-suggest" style="${BTN_CANCEL};width:100%;" title="Häufigste Wort-Korrekturen aus dem Edit-Log (Diktat-bestätigen-Modus)">💡 Vorschläge aus Korrekturen</button>
            <div id="schmidi-words-suggest-list" style="display:none;max-height:120px;overflow-y:auto;border:1px solid #333;border-radius:6px;"></div>
            <div id="schmidi-words-status" style="color:#aaa;font-size:11px;min-height:14px;"></div>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="schmidi-cancel-woerter" style="${BTN_CANCEL}">Schließen</button>
                <button id="schmidi-save-words"     style="${BTN_PRIMARY}">Speichern</button>
            </div>
        </div>

        <!-- Aufnehmen tab -->
        <div id="schmidi-pane-aufnehmen" style="display:none;flex-direction:column;gap:6px;max-width:340px;">
            <div style="display:flex;gap:6px;align-items:center;">
                <button id="schmidi-prev-sentence" style="${BTN_CANCEL}">◀</button>
                <span id="schmidi-sentence-index" style="${LABEL_STYLE}">0/0</span>
                <button id="schmidi-next-sentence" style="${BTN_CANCEL}">▶</button>
            </div>
            <div id="schmidi-training-sentence" style="background:#2a2a2a;border:1px solid #555;border-radius:6px;padding:8px;font-size:13px;line-height:1.4;min-height:40px;color:#eee;">Lade...</div>
            <div id="schmidi-edit-area" style="display:none;flex-direction:column;gap:4px;">
                <input id="schmidi-edit-input" type="text" style="${INPUT_STYLE}"/>
                <div style="display:flex;gap:6px;">
                    <button id="schmidi-edit-save"   style="${BTN_PRIMARY};flex:1;">Speichern</button>
                    <button id="schmidi-edit-cancel" style="${BTN_CANCEL};flex:1;">Abbrechen</button>
                </div>
            </div>
            <div id="schmidi-add-area" style="display:none;flex-direction:column;gap:4px;">
                <input id="schmidi-add-input" type="text" style="${INPUT_STYLE}" placeholder="Neuer Kalibrierungssatz…"/>
                <div style="display:flex;gap:6px;">
                    <button id="schmidi-add-confirm" style="${BTN_PRIMARY};flex:1;">Hinzufügen</button>
                    <button id="schmidi-add-cancel"  style="${BTN_CANCEL};flex:1;">Abbrechen</button>
                </div>
            </div>
            <div style="display:flex;gap:6px;">
                <button id="schmidi-sentence-edit"   style="${BTN_CANCEL};flex:1;">✏ Bearbeiten</button>
                <button id="schmidi-sentence-add"    style="${BTN_CANCEL};flex:1;">+ Neu</button>
                <button id="schmidi-sentence-remove" style="${BTN_CANCEL};flex:1;">✕ Entfernen</button>
            </div>
            <div style="display:flex;gap:6px;">
                <button id="schmidi-training-record"   style="${BTN_PRIMARY};flex:1;">⏺ Aufnehmen</button>
                <button id="schmidi-training-vorhören" style="${BTN_CANCEL};flex:1;" disabled>▶ Vorhören</button>
            </div>
            <button id="schmidi-training-save" style="${BTN_CANCEL};width:100%;" disabled>💾 Speichern</button>
            <div id="schmidi-aufnehmen-status" style="${LABEL_STYLE};min-height:14px;"></div>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="schmidi-cancel-aufnehmen" style="${BTN_CANCEL}">Schließen</button>
            </div>
        </div>

        <!-- Paare tab -->
        <div id="schmidi-pane-training" style="display:none;flex-direction:column;gap:6px;max-width:340px;">
            <div id="schmidi-pairs-list" style="max-height:200px;overflow-y:auto;border:1px solid #333;border-radius:6px;">
                <div style="padding:8px;color:#666;font-size:12px;">Lade…</div>
            </div>
            <div id="schmidi-training-count" style="${LABEL_STYLE}">— Paare</div>
            <div style="border-top:1px solid #444;margin:4px 0;"></div>
            <label style="display:flex;align-items:center;gap:8px;${LABEL_STYLE}">
                <input type="checkbox" id="schmidi-lora-enabled" />
                LoRA verwenden (aktivieren = laden, deaktivieren = entladen)
            </label>
            <button id="schmidi-training-run"  style="${BTN_PRIMARY}">▶ LoRA trainieren</button>
            <button id="schmidi-lora-reload"   style="${BTN_CANCEL}">↺ LoRA neu laden</button>
            <div id="schmidi-training-status" style="${LABEL_STYLE};min-height:14px;"></div>
            <pre id="schmidi-training-log" style="background:#111;border:1px solid #333;border-radius:4px;padding:6px;font-size:10px;max-height:80px;overflow-y:auto;margin:0;display:none;color:#8f8;"></pre>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="schmidi-cancel-training" style="${BTN_CANCEL}">Schließen</button>
            </div>
        </div>

        <!-- Diktate tab (real dictations → review → training pairs) -->
        <div id="schmidi-pane-diktate" style="display:none;flex-direction:column;gap:6px;max-width:340px;">
            <button id="schmidi-dikt-save-last" style="${BTN_PRIMARY}" disabled>💾 Letztes Diktat speichern</button>
            <div id="schmidi-dikt-last-info" style="${LABEL_STYLE}">Kein Diktat in dieser Sitzung</div>
            <div id="schmidi-dikt-list" style="max-height:140px;overflow-y:auto;border:1px solid #333;border-radius:6px;">
                <div style="padding:8px;color:#666;font-size:12px;">Lade…</div>
            </div>
            <div id="schmidi-dikt-detail" style="display:none;flex-direction:column;gap:6px;">
                <textarea id="schmidi-dikt-text" rows="4" style="${INPUT_STYLE}resize:vertical;"></textarea>
                <div style="display:flex;gap:6px;">
                    <button id="schmidi-dikt-transcribe-voxtral" style="${BTN_CANCEL};flex:1;" title="Audio mit Voxtral neu transkribieren">↻ Voxtral</button>
                    <button id="schmidi-dikt-transcribe-qwen"    style="${BTN_CANCEL};flex:1;" title="Audio mit Qwen3 neu transkribieren">↻ Qwen3</button>
                </div>
                <button id="schmidi-dikt-accept" style="${BTN_PRIMARY}">✓ Als Trainingspaar übernehmen</button>
            </div>
            <div id="schmidi-dikt-status" style="${LABEL_STYLE};min-height:14px;"></div>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="schmidi-dikt-close" style="${BTN_CANCEL}">Schließen</button>
            </div>
        </div>

        <!-- Einstellungen tab (Client + Server merged, last) -->
        <div id="schmidi-pane-einstellungen" style="display:none;flex-direction:column;gap:6px;">
            <div style="${LABEL_STYLE}">Server URL</div>
            <input id="schmidi-url-input" type="text" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE}">API-Schlüssel</div>
            <input id="schmidi-apikey-input" type="text" style="${INPUT_STYLE}" placeholder="aus dem Server-Log kopieren"/>
            <div style="${LABEL_STYLE}">Tastenkürzel (z.B. ctrl+shift+d, alt+s)</div>
            <input id="schmidi-hotkey-input" type="text" style="${INPUT_STYLE}"/>
            <div style="border-top:1px solid #333;margin:2px 0;"></div>
            <div id="schmidi-row-delay" style="display:none;flex-direction:column;gap:6px;">
                <div style="${LABEL_STYLE}">Delay tokens (1–30, je 80ms)</div>
                <input id="schmidi-delay" type="number" min="1" max="30" style="${INPUT_STYLE}"/>
            </div>
            <div style="${LABEL_STYLE}">Silence threshold (RMS)</div>
            <input id="schmidi-silence-threshold" type="number" step="0.001" min="0" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE}">Silence detection chunks (je 80ms)</div>
            <input id="schmidi-silence-flush" type="number" min="1" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE}">Min speech chunks (je 80ms)</div>
            <input id="schmidi-min-speech" type="number" min="1" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE}">RMS EMA alpha (0.0–1.0)</div>
            <input id="schmidi-rms-ema" type="number" step="0.01" min="0" max="1" style="${INPUT_STYLE}"/>
            <div style="display:flex;align-items:center;gap:8px;margin-top:2px;">
                <input type="checkbox" id="schmidi-fuzzy-hotwords" style="cursor:pointer;"/>
                <label for="schmidi-fuzzy-hotwords" style="${LABEL_STYLE}cursor:pointer;">Fuzzy-Korrektur eigener Wörter (phonetisch)</label>
            </div>
            <div style="${LABEL_STYLE}">Fuzzy max ratio (0.0–1.0, kleiner = strenger)</div>
            <input id="schmidi-fuzzy-max-ratio" type="number" step="0.01" min="0" max="1" style="${INPUT_STYLE}"/>
            <div id="schmidi-row-german-prime" style="display:none;align-items:center;gap:8px;margin-top:2px;">
                <input type="checkbox" id="schmidi-german-prime" style="cursor:pointer;"/>
                <label for="schmidi-german-prime" style="${LABEL_STYLE}cursor:pointer;">German-Priming (experimentell, Sprach-Bias)</label>
            </div>
            <div id="schmidi-row-context-biasing" style="display:none;align-items:center;gap:8px;margin-top:2px;">
                <input type="checkbox" id="schmidi-context-biasing" style="cursor:pointer;"/>
                <label for="schmidi-context-biasing" style="${LABEL_STYLE}cursor:pointer;">Kontext-Biasing (Hotwords im System-Prompt)</label>
            </div>
            <div style="display:flex;align-items:center;gap:8px;margin-top:2px;">
                <input type="checkbox" id="schmidi-commit-mode" style="cursor:pointer;"/>
                <label for="schmidi-commit-mode" style="${LABEL_STYLE}cursor:pointer;">Diktat bestätigen (↵) statt sofort einfügen</label>
            </div>
            <div id="schmidi-einstellungen-status" style="color:#aaa;font-size:11px;min-height:14px;"></div>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="schmidi-cancel-einstellungen" style="${BTN_CANCEL}">Abbrechen</button>
                <button id="schmidi-save-einstellungen"   style="${BTN_PRIMARY}">Speichern</button>
            </div>
        </div>
    `;
    document.body.appendChild(configPanel);

    // ---- Backend switching ----
    let currentTab = "einstellungen";

    function styleBackendButtons() {
        const active = activeBackend();
        configPanel.querySelectorAll("[data-backend]").forEach((b) => {
            const isActive = b.dataset.backend === active;
            b.style.background = isActive ? "#2980b9" : "#333";
            b.style.color      = isActive ? "#fff" : "#aaa";
        });
        const vEl = configPanel.querySelector("#schmidi-server-version");
        if (vEl) vEl.textContent = `[${BACKENDS[active].label}]`;
    }

    // Reload whatever the current tab shows from the newly selected backend.
    const TAB_LOADERS = {
        woerter: () => loadWords(),
        hotwords: () => loadHotwords(),
        aufnehmen: () => loadTrainingSentences(),
        zweiterdurchgang: () => loadSecondPassSentences(),
        training: () => loadTrainingPairs(),
        diktate: () => loadReviews(),
        einstellungen: () => { fillClientFields(); loadServerParams(); },
    };

    function switchBackend(b) {
        if (recording) { showToast("Serverwechsel während der Aufnahme nicht möglich"); return; }
        GM_setValue("active_backend", b);
        styleBackendButtons();
        updateBtnTitle();
        const loader = TAB_LOADERS[currentTab];
        if (loader) loader();
    }

    configPanel.querySelectorAll("[data-backend]").forEach((b) => {
        b.addEventListener("click", () => switchBackend(b.dataset.backend));
    });

    // ---- Tab switching ----
    function switchTab(tab) {
        currentTab = tab;
        ["woerter", "hotwords", "aufnehmen", "zweiterdurchgang", "training", "diktate", "einstellungen"].forEach((t) => {
            const pane   = configPanel.querySelector(`#schmidi-pane-${t}`);
            const tabBtn = configPanel.querySelector(`#schmidi-tab-${t}`);
            if (!pane || !tabBtn) return;
            const active = t === tab;
            pane.style.display         = active ? "flex" : "none";
            tabBtn.style.borderBottomColor = active ? "#2980b9" : "transparent";
            tabBtn.style.color         = active ? "#ddd" : "#888";
            tabBtn.style.fontWeight    = active ? "bold" : "normal";
        });
    }
    configPanel.querySelector("#schmidi-tab-woerter").addEventListener("click", () => { switchTab("woerter"); loadWords(); });
    configPanel.querySelector("#schmidi-tab-hotwords").addEventListener("click", () => { switchTab("hotwords"); loadHotwords(); });
    configPanel.querySelector("#schmidi-tab-aufnehmen").addEventListener("click", () => { switchTab("aufnehmen"); loadTrainingSentences(); });
    configPanel.querySelector("#schmidi-tab-zweiterdurchgang").addEventListener("click", () => { switchTab("zweiterdurchgang"); loadSecondPassSentences(); });
    configPanel.querySelector("#schmidi-tab-training").addEventListener("click", () => { switchTab("training"); loadTrainingPairs(); });
    configPanel.querySelector("#schmidi-tab-diktate").addEventListener("click", () => { switchTab("diktate"); loadReviews(); });
    configPanel.querySelector("#schmidi-tab-einstellungen").addEventListener("click", () => { switchTab("einstellungen"); fillClientFields(); loadServerParams(); });

    // ---- HTTP helpers ----
    function getHttpBase() {
        return getServerUrl()
            .replace(/^ws(s?):\/\//, "http$1://")
            .replace(/\/asr$/, "");
    }

    function setWordsStatus(msg, isError) {
        const el = configPanel.querySelector("#schmidi-words-status");
        if (el) { el.textContent = msg; el.style.color = isError ? "#e74c3c" : "#aaa"; }
    }

    function setEinstellungenStatus(msg, isError) {
        const el = configPanel.querySelector("#schmidi-einstellungen-status");
        if (el) { el.textContent = msg; el.style.color = isError ? "#e74c3c" : "#aaa"; }
    }

    function setHotwordsStatus(msg, isError) {
        const el = configPanel.querySelector("#schmidi-hotwords-status");
        if (el) { el.textContent = msg; el.style.color = isError ? "#e74c3c" : "#aaa"; }
    }

    // Hotwords live in GM storage and are appended to the WS URL per session —
    // no server round-trip needed to load/save.
    function loadHotwords() {
        configPanel.querySelector("#schmidi-hotwords").value = getHotwords();
        setHotwordsStatus("", false);
    }

    function saveHotwords() {
        setHotwords(configPanel.querySelector("#schmidi-hotwords").value);
        setHotwordsStatus("Gespeichert ✓", false);
    }

    // Fill the client-side fields (URL, API key, hotkey, commit mode) for the
    // active backend — called on panel open and after a backend switch.
    function fillClientFields() {
        configPanel.querySelector("#schmidi-url-input").value    = getServerUrl();
        configPanel.querySelector("#schmidi-apikey-input").value = apiKey();
        configPanel.querySelector("#schmidi-hotkey-input").value = getHotkey();
        configPanel.querySelector("#schmidi-commit-mode").checked = isCommitMode();
    }

    // ---- Load functions ----

    async function loadWords() {
        setWordsStatus("Lade Wörter…", false);
        try {
            const res = await authFetch(`${getHttpBase()}/words`);
            if (!res.ok) throw new Error("GET /words: " + res.status);
            const data = await res.json();
            configPanel.querySelector("#schmidi-words").value = (data.words || []).join("\n");
            setWordsStatus("", false);
        } catch (e) {
            setWordsStatus("Fehler: " + e.message, true);
        }
    }

    async function loadServerParams() {
        setEinstellungenStatus("Lade Einstellungen…", false);
        try {
            const res = await authFetch(`${getHttpBase()}/config`);
            if (!res.ok) throw new Error("GET /config: " + res.status);
            const cfg = await res.json();
            lastCfg = cfg;
            configPanel.querySelector("#schmidi-delay").value             = cfg.delay ?? "";
            configPanel.querySelector("#schmidi-silence-threshold").value = cfg.silence_threshold ?? "";
            configPanel.querySelector("#schmidi-silence-flush").value     = cfg.silence_flush ?? "";
            configPanel.querySelector("#schmidi-min-speech").value        = cfg.min_speech ?? "";
            configPanel.querySelector("#schmidi-rms-ema").value           = cfg.rms_ema ?? "";
            configPanel.querySelector("#schmidi-fuzzy-hotwords").checked  = cfg.fuzzy_hotwords ?? true;
            configPanel.querySelector("#schmidi-fuzzy-max-ratio").value   = cfg.fuzzy_max_ratio ?? "";
            configPanel.querySelector("#schmidi-german-prime").checked    = cfg.german_prime ?? false;
            configPanel.querySelector("#schmidi-context-biasing").checked = cfg.context_biasing !== false;
            // Show only the settings the connected backend actually reports.
            configPanel.querySelector("#schmidi-row-delay").style.display =
                cfg.delay !== undefined ? "flex" : "none";
            configPanel.querySelector("#schmidi-row-german-prime").style.display =
                cfg.german_prime !== undefined ? "flex" : "none";
            configPanel.querySelector("#schmidi-row-context-biasing").style.display =
                cfg.context_biasing !== undefined ? "flex" : "none";
            const vEl = configPanel.querySelector("#schmidi-server-version");
            if (vEl) {
                vEl.textContent = `[${BACKENDS[activeBackend()].label}]` +
                    (cfg.version ? ` v${cfg.version}` : "");
            }
            setEinstellungenStatus("", false);
        } catch (e) {
            setEinstellungenStatus("Fehler: " + e.message, true);
        }
    }

    // ---- Eigene Wörter: suggestions mined from the edit log ----
    // GET /edits/report aggregates commit-mode original→edited diffs into the
    // most frequent word-level corrections; ＋ appends a wrong=correct line to
    // the textarea (the user still saves explicitly).
    async function loadWordSuggestions() {
        const listEl = configPanel.querySelector('#schmidi-words-suggest-list');
        if (!listEl) return;
        listEl.style.display = 'block';
        listEl.innerHTML = '<div style="padding:6px;color:#666;font-size:11px;">Lade…</div>';
        try {
            const res = await authFetch(`${getHttpBase()}/edits/report`);
            if (!res.ok) throw new Error(res.status === 404
                ? 'Auf diesem Server nicht verfügbar' : 'GET /edits/report: ' + res.status);
            const data = await res.json();
            const sugg = data.suggestions || [];
            if (sugg.length === 0) {
                listEl.innerHTML = `<div style="padding:6px;color:#666;font-size:11px;">Keine Korrekturen im Edit-Log (${data.entries || 0} Diktate)</div>`;
                return;
            }
            listEl.innerHTML = sugg.map(s => `
                <div style="display:flex;align-items:center;gap:4px;padding:3px 6px;border-bottom:1px solid #2a2a2a;">
                    <span style="flex:1;font-size:11px;color:#ccc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(s.original)} → ${escapeHtml(s.edited)}">${escapeHtml(s.original)} → ${escapeHtml(s.edited)}</span>
                    <span style="color:#888;font-size:10px;">${s.count}×</span>
                    <button class="sp-sugg-add" data-line="${escapeHtml(`${s.original}=${s.edited}`)}" style="background:#333;color:#8f8;border:1px solid #555;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:11px;">＋</button>
                </div>`).join('');
            listEl.onclick = (e) => {
                const addBtn = e.target.closest('.sp-sugg-add');
                if (!addBtn) return;
                const ta   = configPanel.querySelector('#schmidi-words');
                const line = addBtn.dataset.line;
                const lines = ta.value.split('\n').map(l => l.trim());
                if (lines.includes(line)) { setWordsStatus('Bereits vorhanden', false); return; }
                ta.value = ta.value.replace(/\s*$/, '') + (ta.value.trim() ? '\n' : '') + line + '\n';
                setWordsStatus(`Hinzugefügt: ${line} — Speichern nicht vergessen`, false);
            };
        } catch (e) {
            listEl.innerHTML = `<div style="padding:6px;color:#e74c3c;font-size:11px;">${escapeHtml(e.message)}</div>`;
        }
    }
    configPanel.querySelector('#schmidi-words-suggest').addEventListener('click', loadWordSuggestions);

    // Parse a numeric input field. Returns undefined for empty/invalid input,
    // but preserves a legitimate 0 (which `value || undefined` would wrongly drop).
    function numField(sel, parse) {
        const raw = configPanel.querySelector(sel).value.trim();
        if (raw === "") return undefined;
        const n = parse(raw);
        return Number.isFinite(n) ? n : undefined;
    }

    async function saveServerParams() {
        const patch = {
            silence_threshold: numField("#schmidi-silence-threshold", parseFloat),
            silence_flush:     numField("#schmidi-silence-flush", parseInt),
            min_speech:        numField("#schmidi-min-speech", parseInt),
            rms_ema:           numField("#schmidi-rms-ema", parseFloat),
            fuzzy_hotwords:    configPanel.querySelector("#schmidi-fuzzy-hotwords").checked,
            fuzzy_max_ratio:   numField("#schmidi-fuzzy-max-ratio", parseFloat),
        };
        // Backend-specific fields — only sent when the connected server reported them.
        if (lastCfg.delay !== undefined)
            patch.delay = numField("#schmidi-delay", parseInt);
        if (lastCfg.german_prime !== undefined)
            patch.german_prime = configPanel.querySelector("#schmidi-german-prime").checked;
        if (lastCfg.context_biasing !== undefined)
            patch.context_biasing = configPanel.querySelector("#schmidi-context-biasing").checked;
        Object.keys(patch).forEach((k) => patch[k] === undefined && delete patch[k]);
        setEinstellungenStatus("Speichere…", false);
        try {
            const res = await authFetch(`${getHttpBase()}/config`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(patch),
            });
            if (!res.ok) throw new Error(await res.text());
            setEinstellungenStatus("Gespeichert ✓", false);
        } catch (e) {
            setEinstellungenStatus("Fehler: " + e.message, true);
        }
    }

    async function saveWords() {
        const rawLines = configPanel
            .querySelector("#schmidi-words")
            .value.split("\n")
            .map((s) => s.trim())
            .filter(Boolean);

        // Duplicate check (skip comment lines)
        const lhsSet   = new Set();
        const plainSet = new Set();
        for (const line of rawLines) {
            if (line.startsWith("#")) continue;
            if (line.includes("=")) {
                const lhs = line.split("=")[0].trim().toLowerCase();
                if (lhsSet.has(lhs)) {
                    setWordsStatus(`Doppelter Eintrag: "${lhs}" — bitte korrigieren`, true);
                    return;
                }
                lhsSet.add(lhs);
            } else {
                const low = line.toLowerCase();
                if (plainSet.has(low)) {
                    setWordsStatus(`Doppelter Eintrag: "${line}" — bitte korrigieren`, true);
                    return;
                }
                plainSet.add(low);
            }
        }

        setWordsStatus("Speichere Wörter…", false);
        try {
            const wordsRes = await authFetch(`${getHttpBase()}/words`);
            const current  = wordsRes.ok ? (await wordsRes.json()).words || [] : [];
            const newSet   = new Set(rawLines);
            const curSet   = new Set(current);
            const add      = rawLines.filter((w) => !curSet.has(w));
            const remove   = current.filter((w) => !newSet.has(w));
            const res = await authFetch(`${getHttpBase()}/words`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ add, remove }),
            });
            if (!res.ok) throw new Error(await res.text());
            setWordsStatus(`Gespeichert ✓ (${rawLines.length} Einträge)`, false);
        } catch (e) {
            setWordsStatus("Fehler: " + e.message, true);
        }
    }

    async function saveEinstellungen() {
        const url = configPanel.querySelector("#schmidi-url-input").value.trim();
        const ak  = configPanel.querySelector("#schmidi-apikey-input").value.trim();
        const hk  = configPanel.querySelector("#schmidi-hotkey-input").value.trim();
        const cm  = configPanel.querySelector("#schmidi-commit-mode").checked;
        if (url) setServerUrl(url);
        setApiKey(ak);
        if (hk)  setHotkey(hk);
        GM_setValue("commit_mode", cm);
        updateBtnTitle();
        await saveServerParams();
    }

    // ---- Config panel actions ----

    function openConfig() {
        fillClientFields();
        styleBackendButtons();
        switchTab("einstellungen");
        loadServerParams();
        configPanel.style.display = "flex";
    }

    function closeConfig() {
        configPanel.style.display = "none";
    }

    configPanel.querySelector("#schmidi-cancel-woerter").addEventListener("click", closeConfig);
    configPanel.querySelector("#schmidi-cancel-hotwords").addEventListener("click", closeConfig);
    configPanel.querySelector("#schmidi-cancel-aufnehmen").addEventListener("click", closeConfig);
    configPanel.querySelector("#sp2-close").addEventListener("click", closeConfig);
    configPanel.querySelector("#schmidi-cancel-training").addEventListener("click", closeConfig);
    configPanel.querySelector("#schmidi-dikt-close").addEventListener("click", closeConfig);
    configPanel.querySelector("#schmidi-cancel-einstellungen").addEventListener("click", closeConfig);
    configPanel.querySelector("#schmidi-save-words").addEventListener("click", saveWords);
    configPanel.querySelector("#schmidi-save-hotwords").addEventListener("click", saveHotwords);
    configPanel.querySelector("#schmidi-save-einstellungen").addEventListener("click", saveEinstellungen);

    // ---- Aufnehmen / Paare state ----
    let trainingSentences  = [];
    let trainingCurrentIdx = 0;
    let trainingRecording  = false;
    let trainingPcmBuffers = [];
    let trainingAudioCtx   = null;
    let trainingScriptProc = null;
    let trainingMicStream  = null;
    let trainingPreviewCtx  = null;
    let trainingPreviewSrc  = null;
    let paaresCurrentAudio  = null;
    let paaresCurrentPlayId = null;
    let trainingPairs      = [];
    let trainingStatusPoll = null;

    function setAufnehmenStatus(msg, isError) {
        const el = configPanel.querySelector('#schmidi-aufnehmen-status');
        if (el) { el.textContent = msg; el.style.color = isError ? '#e74c3c' : '#aaa'; }
    }
    function setPaareStatus(msg, isError) {
        const el = configPanel.querySelector('#schmidi-training-status');
        if (el) { el.textContent = msg; el.style.color = isError ? '#e74c3c' : '#aaa'; }
    }

    // Prefer a 16 kHz capture context: the browser then resamples the mic
    // stream itself with proper low-pass filtering (same as the preview player).
    // The old path decimated the native 48 kHz stream by picking every 3rd
    // sample, which folds high frequencies into the speech band as aliasing
    // noise. Naive decimation remains only as a fallback for browsers that
    // refuse a fixed-rate AudioContext.
    function createCaptureContext() {
        const AC = window.AudioContext || window.webkitAudioContext;
        try { return new AC({ sampleRate: SAMPLE_RATE }); }
        catch (_) { return new AC(); }
    }

    // 16 kHz mono copy of one ScriptProcessor input buffer: pass-through copy
    // when the context already runs at 16 kHz, decimation fallback otherwise.
    function captureChunk(input, ratio) {
        if (ratio === 1) return new Float32Array(input);
        const outLen = Math.floor(input.length / ratio);
        const out = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) out[i] = input[Math.floor(i * ratio)];
        return out;
    }

    // Flatten the recorded chunk list into one contiguous Float32Array.
    function concatPcm(buffers) {
        const totalLen = buffers.reduce((s, b) => s + b.length, 0);
        const combined = new Float32Array(totalLen);
        let offset = 0;
        for (const buf of buffers) { combined.set(buf, offset); offset += buf.length; }
        return combined;
    }

    // ---- Aufnehmen: sentence navigation ----

    async function loadTrainingSentences() {
        try {
            const res = await authFetch(`${getHttpBase()}/training/sentences`);
            if (!res.ok) throw new Error('GET /training/sentences: ' + res.status);
            const data = await res.json();
            trainingSentences = (data.sentences || []).filter(s => !s.recorded);
            trainingCurrentIdx = Math.min(trainingCurrentIdx, Math.max(0, trainingSentences.length - 1));
            renderCurrentSentence();
        } catch (e) {
            setAufnehmenStatus('Fehler: ' + e.message, true);
        }
    }

    function renderCurrentSentence() {
        const sentEl = configPanel.querySelector('#schmidi-training-sentence');
        const idxEl  = configPanel.querySelector('#schmidi-sentence-index');
        if (!sentEl || !idxEl) return;
        const total = trainingSentences.length;
        if (total === 0) {
            sentEl.textContent = 'Keine Sätze verfügbar';
            idxEl.textContent  = '0/0';
            return;
        }
        const entry = trainingSentences[trainingCurrentIdx] || {};
        sentEl.textContent       = entry.text || '';
        sentEl.style.borderColor = '#555';
        sentEl.style.background  = '#2a2a2a';
        idxEl.textContent = `${trainingCurrentIdx + 1}/${total}`;
    }

    configPanel.querySelector('#schmidi-prev-sentence').addEventListener('click', () => {
        if (!trainingSentences.length) return;
        trainingCurrentIdx = (trainingCurrentIdx - 1 + trainingSentences.length) % trainingSentences.length;
        renderCurrentSentence();
    });
    configPanel.querySelector('#schmidi-next-sentence').addEventListener('click', () => {
        if (!trainingSentences.length) return;
        trainingCurrentIdx = (trainingCurrentIdx + 1) % trainingSentences.length;
        renderCurrentSentence();
    });

    // ---- Aufnehmen: sentence edit ----

    function startEditSentence() {
        const entry     = trainingSentences[trainingCurrentIdx] || {};
        const editArea  = configPanel.querySelector('#schmidi-edit-area');
        const editInput = configPanel.querySelector('#schmidi-edit-input');
        const sentEl    = configPanel.querySelector('#schmidi-training-sentence');
        if (editArea)  editArea.style.display = 'flex';
        if (sentEl)    sentEl.style.display   = 'none';
        if (editInput) { editInput.value = entry.text || ''; editInput.focus(); editInput.select(); }
    }
    function cancelEditSentence() {
        const editArea = configPanel.querySelector('#schmidi-edit-area');
        const sentEl   = configPanel.querySelector('#schmidi-training-sentence');
        if (editArea) editArea.style.display = 'none';
        if (sentEl)   sentEl.style.display   = 'block';
    }
    async function saveEditedSentence() {
        const entry     = trainingSentences[trainingCurrentIdx] || {};
        const oldText   = entry.text || '';
        const editInput = configPanel.querySelector('#schmidi-edit-input');
        const newText   = (editInput ? editInput.value : '').trim();
        if (!newText) { setAufnehmenStatus('Leerer Text', true); return; }
        if (newText === oldText) { cancelEditSentence(); return; }
        try {
            const res = await authFetch(`${getHttpBase()}/training/sentence`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ old: oldText, new: newText }),
            });
            if (!res.ok) throw new Error(await res.text());
            cancelEditSentence();
            setAufnehmenStatus('Satz aktualisiert', false);
            await loadTrainingSentences();
        } catch (e) {
            setAufnehmenStatus('Fehler: ' + e.message, true);
        }
    }

    configPanel.querySelector('#schmidi-sentence-edit').addEventListener('click', startEditSentence);
    configPanel.querySelector('#schmidi-edit-save').addEventListener('click', saveEditedSentence);
    configPanel.querySelector('#schmidi-edit-cancel').addEventListener('click', cancelEditSentence);

    // ---- Aufnehmen: add sentence ----

    function startAddSentence() {
        const addArea  = configPanel.querySelector('#schmidi-add-area');
        const addInput = configPanel.querySelector('#schmidi-add-input');
        if (addArea)  addArea.style.display = 'flex';
        if (addInput) { addInput.value = ''; addInput.focus(); }
    }
    function cancelAddSentence() {
        const addArea = configPanel.querySelector('#schmidi-add-area');
        if (addArea) addArea.style.display = 'none';
    }
    async function confirmAddSentence() {
        const addInput = configPanel.querySelector('#schmidi-add-input');
        const text     = (addInput ? addInput.value : '').trim();
        if (!text) { setAufnehmenStatus('Leerer Satz', true); return; }
        try {
            const res = await authFetch(`${getHttpBase()}/training/sentence`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            if (!res.ok) throw new Error(await res.text());
            cancelAddSentence();
            setAufnehmenStatus('Satz hinzugefügt', false);
            await loadTrainingSentences();
            trainingCurrentIdx = Math.max(0, trainingSentences.length - 1);
            renderCurrentSentence();
        } catch (e) {
            setAufnehmenStatus('Fehler: ' + e.message, true);
        }
    }

    configPanel.querySelector('#schmidi-sentence-add').addEventListener('click', startAddSentence);
    configPanel.querySelector('#schmidi-add-confirm').addEventListener('click', confirmAddSentence);
    configPanel.querySelector('#schmidi-add-cancel').addEventListener('click', cancelAddSentence);

    // ---- Aufnehmen: delete sentence ----

    async function deleteCurrentSentence() {
        const entry = trainingSentences[trainingCurrentIdx] || {};
        const text  = (entry.text || '').trim();
        if (!text) return;
        try {
            const res = await authFetch(`${getHttpBase()}/training/sentence`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            if (!res.ok) throw new Error(await res.text());
            trainingCurrentIdx = Math.max(0, Math.min(trainingCurrentIdx, trainingSentences.length - 2));
            setAufnehmenStatus('Satz entfernt', false);
            loadTrainingSentences();
        } catch (e) {
            setAufnehmenStatus('Fehler: ' + e.message, true);
        }
    }

    configPanel.querySelector('#schmidi-sentence-remove').addEventListener('click', deleteCurrentSentence);

    // ---- Aufnehmen: recording ----

    configPanel.querySelector('#schmidi-training-record').addEventListener('click', () => {
        if (trainingRecording) stopTrainingRecording();
        else startTrainingRecording();
    });
    configPanel.querySelector('#schmidi-training-vorhören').addEventListener('click', previewRecording);
    configPanel.querySelector('#schmidi-training-save').addEventListener('click', saveTrainingPair);

    function startTrainingRecording() {
        if (trainingPreviewSrc) { try { trainingPreviewSrc.stop(); } catch(_) {} trainingPreviewSrc = null; }
        if (trainingPreviewCtx) { trainingPreviewCtx.close(); trainingPreviewCtx = null; }
        trainingPcmBuffers = [];
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then((stream) => {
                trainingMicStream = stream;
                trainingRecording = true;
                trainingAudioCtx = createCaptureContext();
                const source = trainingAudioCtx.createMediaStreamSource(stream);
                const ratio  = trainingAudioCtx.sampleRate / SAMPLE_RATE;
                const proc   = trainingAudioCtx.createScriptProcessor(4096, 1, 1);
                source.connect(proc);
                // Silent gain node: keeps the graph alive without playing the mic
                // through the speakers (direct destination connect triggers browser
                // AEC self-cancellation, which silences the recorded speech).
                const silentOutT = trainingAudioCtx.createGain();
                silentOutT.gain.value = 0;
                proc.connect(silentOutT);
                silentOutT.connect(trainingAudioCtx.destination);
                proc.onaudioprocess = (e) => {
                    if (!trainingRecording) return;
                    trainingPcmBuffers.push(captureChunk(e.inputBuffer.getChannelData(0), ratio));
                };
                trainingScriptProc = proc;
                setAufnehmenStatus('Aufnahme…', false);
                const recBtn = configPanel.querySelector('#schmidi-training-record');
                if (recBtn) { recBtn.textContent = '⏹ Stop'; recBtn.style.background = '#c0392b'; }
            })
            .catch((err) => setAufnehmenStatus('Mikrofon: ' + err.message, true));
    }

    function stopTrainingRecording() {
        trainingRecording = false;
        if (trainingScriptProc) { trainingScriptProc.disconnect(); trainingScriptProc = null; }
        if (trainingAudioCtx)   { trainingAudioCtx.close(); trainingAudioCtx = null; }
        if (trainingMicStream)  { trainingMicStream.getTracks().forEach(t => t.stop()); trainingMicStream = null; }
        finalizeTrainingRecording();
    }

    function finalizeTrainingRecording() {
        const recBtn      = configPanel.querySelector('#schmidi-training-record');
        const saveBtn     = configPanel.querySelector('#schmidi-training-save');
        const vorhörenBtn = configPanel.querySelector('#schmidi-training-vorhören');
        if (recBtn) { recBtn.textContent = '⏺ Aufnehmen'; recBtn.style.background = '#2980b9'; }
        const hasAudio = trainingPcmBuffers.length > 0;
        if (saveBtn)     saveBtn.disabled     = !hasAudio;
        if (vorhörenBtn) vorhörenBtn.disabled = !hasAudio;
        setAufnehmenStatus(hasAudio ? 'Aufnahme beendet' : 'Keine Daten', !hasAudio);
    }

    function previewRecording() {
        const vorhörenBtn = configPanel.querySelector('#schmidi-training-vorhören');
        if (trainingPreviewSrc) {
            try { trainingPreviewSrc.stop(); } catch(_) {}
            trainingPreviewSrc = null;
            if (trainingPreviewCtx) { trainingPreviewCtx.close(); trainingPreviewCtx = null; }
            if (vorhörenBtn) vorhörenBtn.textContent = '▶ Vorhören';
            setAufnehmenStatus('', false);
            return;
        }
        if (trainingPcmBuffers.length === 0) return;
        const combined = concatPcm(trainingPcmBuffers);
        try {
            trainingPreviewCtx = new AudioContext({ sampleRate: 16000 });
            const ctx      = trainingPreviewCtx;
            const audioBuf = ctx.createBuffer(1, combined.length, 16000);
            audioBuf.copyToChannel(combined, 0);
            trainingPreviewSrc = ctx.createBufferSource();
            const src = trainingPreviewSrc;
            src.buffer = audioBuf;
            src.connect(ctx.destination);
            src.start();
            if (vorhörenBtn) vorhörenBtn.textContent = '⏹ Stopp';
            setAufnehmenStatus('Wiedergabe…', false);
            src.onended = () => {
                ctx.close(); trainingPreviewCtx = null; trainingPreviewSrc = null;
                if (vorhörenBtn) vorhörenBtn.textContent = '▶ Vorhören';
                setAufnehmenStatus('', false);
            };
        } catch (e) {
            setAufnehmenStatus('Wiedergabe: ' + e.message, true);
        }
    }

    async function saveTrainingPair() {
        if (trainingPcmBuffers.length === 0) return;
        const entry = trainingSentences[trainingCurrentIdx] || {};
        const text  = (entry.text || '').trim();
        if (!text) { setAufnehmenStatus('Kein Satz ausgewählt', true); return; }

        const combined = concatPcm(trainingPcmBuffers);

        setAufnehmenStatus('Speichere…', false);
        try {
            const res = await authFetch(
                `${getHttpBase()}/training/pair?text=${encodeURIComponent(text)}`,
                { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: combined.buffer }
            );
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            setAufnehmenStatus(`Gespeichert ✓ (Paar ${data.id}, ${data.duration_s.toFixed(1)}s)`, false);
            trainingPcmBuffers = [];
            const saveBtn     = configPanel.querySelector('#schmidi-training-save');
            const vorhörenBtn = configPanel.querySelector('#schmidi-training-vorhören');
            if (saveBtn)     saveBtn.disabled     = true;
            if (vorhörenBtn) vorhörenBtn.disabled = true;
            if (trainingSentences.length > 1) {
                trainingCurrentIdx = (trainingCurrentIdx + 1) % trainingSentences.length;
            }
            loadTrainingSentences();
        } catch (e) {
            setAufnehmenStatus('Fehler: ' + e.message, true);
        }
    }

    // ---- 2. Durchgang tab (second recording pass over once-recorded sentences) ----

    let secondPassSentences = [];
    let sp2CurrentIdx       = 0;
    let sp2Recording        = false;
    let sp2PcmBuffers       = [];
    let sp2AudioCtx         = null;
    let sp2ScriptProc       = null;
    let sp2MicStream        = null;
    let sp2PreviewCtx       = null;
    let sp2PreviewSrc       = null;

    function setSp2Status(msg, isError) {
        const el = configPanel.querySelector('#sp2-status');
        if (el) { el.textContent = msg; el.style.color = isError ? '#e74c3c' : '#aaa'; }
    }

    async function loadSecondPassSentences() {
        try {
            const res = await authFetch(`${getHttpBase()}/training/sentences`);
            if (!res.ok) throw new Error('GET /training/sentences: ' + res.status);
            const data = await res.json();
            secondPassSentences = (data.sentences || []).filter(s => s.pair_ids && s.pair_ids.length === 1);
            sp2CurrentIdx = Math.min(sp2CurrentIdx, Math.max(0, secondPassSentences.length - 1));
            renderSp2Sentence();
        } catch (e) {
            setSp2Status('Fehler: ' + e.message, true);
        }
    }

    function renderSp2Sentence() {
        const sentEl = configPanel.querySelector('#sp2-sentence');
        const idxEl  = configPanel.querySelector('#sp2-index');
        if (!sentEl || !idxEl) return;
        const total = secondPassSentences.length;
        if (total === 0) {
            sentEl.textContent = 'Alle Sätze fertig ✓';
            idxEl.textContent  = '0/0';
            return;
        }
        const entry = secondPassSentences[sp2CurrentIdx] || {};
        sentEl.textContent = entry.text || '';
        idxEl.textContent  = `${sp2CurrentIdx + 1}/${total}`;
    }

    configPanel.querySelector('#sp2-prev').addEventListener('click', () => {
        if (!secondPassSentences.length) return;
        sp2CurrentIdx = (sp2CurrentIdx - 1 + secondPassSentences.length) % secondPassSentences.length;
        renderSp2Sentence();
    });
    configPanel.querySelector('#sp2-next').addEventListener('click', () => {
        if (!secondPassSentences.length) return;
        sp2CurrentIdx = (sp2CurrentIdx + 1) % secondPassSentences.length;
        renderSp2Sentence();
    });

    configPanel.querySelector('#sp2-record').addEventListener('click', () => {
        if (sp2Recording) stopSp2Recording();
        else startSp2Recording();
    });
    configPanel.querySelector('#sp2-vorhören').addEventListener('click', previewSp2Recording);
    configPanel.querySelector('#sp2-save').addEventListener('click', saveSp2Pair);

    function startSp2Recording() {
        if (sp2PreviewSrc) { try { sp2PreviewSrc.stop(); } catch(_) {} sp2PreviewSrc = null; }
        if (sp2PreviewCtx) { sp2PreviewCtx.close(); sp2PreviewCtx = null; }
        sp2PcmBuffers = [];
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then((stream) => {
                sp2MicStream = stream;
                sp2Recording = true;
                sp2AudioCtx  = createCaptureContext();
                const source = sp2AudioCtx.createMediaStreamSource(stream);
                const ratio  = sp2AudioCtx.sampleRate / SAMPLE_RATE;
                const proc   = sp2AudioCtx.createScriptProcessor(4096, 1, 1);
                source.connect(proc);
                const silentOut = sp2AudioCtx.createGain();
                silentOut.gain.value = 0;
                proc.connect(silentOut);
                silentOut.connect(sp2AudioCtx.destination);
                proc.onaudioprocess = (e) => {
                    if (!sp2Recording) return;
                    sp2PcmBuffers.push(captureChunk(e.inputBuffer.getChannelData(0), ratio));
                };
                sp2ScriptProc = proc;
                setSp2Status('Aufnahme…', false);
                const recBtn = configPanel.querySelector('#sp2-record');
                if (recBtn) { recBtn.textContent = '⏹ Stop'; recBtn.style.background = '#c0392b'; }
            })
            .catch((err) => setSp2Status('Mikrofon: ' + err.message, true));
    }

    function stopSp2Recording() {
        sp2Recording = false;
        if (sp2ScriptProc) { sp2ScriptProc.disconnect(); sp2ScriptProc = null; }
        if (sp2AudioCtx)   { sp2AudioCtx.close(); sp2AudioCtx = null; }
        if (sp2MicStream)  { sp2MicStream.getTracks().forEach(t => t.stop()); sp2MicStream = null; }
        finalizeSp2Recording();
    }

    function finalizeSp2Recording() {
        const recBtn      = configPanel.querySelector('#sp2-record');
        const saveBtn     = configPanel.querySelector('#sp2-save');
        const vorhörenBtn = configPanel.querySelector('#sp2-vorhören');
        if (recBtn) { recBtn.textContent = '⏺ Aufnehmen'; recBtn.style.background = '#2980b9'; }
        const hasAudio = sp2PcmBuffers.length > 0;
        if (saveBtn)     saveBtn.disabled     = !hasAudio;
        if (vorhörenBtn) vorhörenBtn.disabled = !hasAudio;
        setSp2Status(hasAudio ? 'Aufnahme beendet' : 'Keine Daten', !hasAudio);
    }

    function previewSp2Recording() {
        const vorhörenBtn = configPanel.querySelector('#sp2-vorhören');
        if (sp2PreviewSrc) {
            try { sp2PreviewSrc.stop(); } catch(_) {}
            sp2PreviewSrc = null;
            if (sp2PreviewCtx) { sp2PreviewCtx.close(); sp2PreviewCtx = null; }
            if (vorhörenBtn) vorhörenBtn.textContent = '▶ Vorhören';
            setSp2Status('', false);
            return;
        }
        if (sp2PcmBuffers.length === 0) return;
        const combined = concatPcm(sp2PcmBuffers);
        try {
            sp2PreviewCtx = new AudioContext({ sampleRate: 16000 });
            const ctx      = sp2PreviewCtx;
            const audioBuf = ctx.createBuffer(1, combined.length, 16000);
            audioBuf.copyToChannel(combined, 0);
            sp2PreviewSrc = ctx.createBufferSource();
            const src     = sp2PreviewSrc;
            src.buffer    = audioBuf;
            src.connect(ctx.destination);
            src.start();
            if (vorhörenBtn) vorhörenBtn.textContent = '⏹ Stopp';
            setSp2Status('Wiedergabe…', false);
            src.onended = () => {
                ctx.close(); sp2PreviewCtx = null; sp2PreviewSrc = null;
                if (vorhörenBtn) vorhörenBtn.textContent = '▶ Vorhören';
                setSp2Status('', false);
            };
        } catch (e) {
            setSp2Status('Wiedergabe: ' + e.message, true);
        }
    }

    async function saveSp2Pair() {
        if (sp2PcmBuffers.length === 0) return;
        const entry = secondPassSentences[sp2CurrentIdx] || {};
        const text  = (entry.text || '').trim();
        if (!text) { setSp2Status('Kein Satz ausgewählt', true); return; }
        const combined = concatPcm(sp2PcmBuffers);
        setSp2Status('Speichere…', false);
        try {
            const res = await authFetch(
                `${getHttpBase()}/training/pair?text=${encodeURIComponent(text)}`,
                { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: combined.buffer }
            );
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            setSp2Status(`Gespeichert ✓ (Paar ${data.id}, ${data.duration_s.toFixed(1)}s)`, false);
            sp2PcmBuffers = [];
            const saveBtn     = configPanel.querySelector('#sp2-save');
            const vorhörenBtn = configPanel.querySelector('#sp2-vorhören');
            if (saveBtn)     saveBtn.disabled     = true;
            if (vorhörenBtn) vorhörenBtn.disabled = true;
            if (secondPassSentences.length > 1) {
                sp2CurrentIdx = (sp2CurrentIdx + 1) % secondPassSentences.length;
            }
            loadSecondPassSentences();
        } catch (e) {
            setSp2Status('Fehler: ' + e.message, true);
        }
    }

    // ---- Paare tab ----

    async function loadTrainingPairs() {
        try {
            const res = await authFetch(`${getHttpBase()}/training/pairs`);
            if (!res.ok) throw new Error('GET /training/pairs: ' + res.status);
            const data = await res.json();
            trainingPairs = data.pairs || [];
            renderPairsList();
            await refreshLoraToggle();
        } catch (e) {
            setPaareStatus('Fehler: ' + e.message, true);
        }
    }

    // Reflect the server's current LoRA state in the "LoRA verwenden" checkbox.
    async function refreshLoraToggle() {
        const box = configPanel.querySelector('#schmidi-lora-enabled');
        if (!box) return;
        try {
            const res = await authFetch(`${getHttpBase()}/config`);
            if (!res.ok) return;
            const cfg = await res.json();
            box.checked = !!cfg.lora_active;
            box.dataset.loraDir = cfg.lora_dir || '';
        } catch (_) { /* leave checkbox as-is on error */ }
    }

    function renderPairsList() {
        if (paaresCurrentAudio) { paaresCurrentAudio.pause(); paaresCurrentAudio.src = ''; paaresCurrentAudio = null; paaresCurrentPlayId = null; }
        const listEl  = configPanel.querySelector('#schmidi-pairs-list');
        const countEl = configPanel.querySelector('#schmidi-training-count');
        if (!listEl) return;
        if (trainingPairs.length === 0) {
            listEl.innerHTML = '<div style="padding:8px;color:#666;font-size:12px;">Keine Aufnahmen</div>';
            if (countEl) countEl.textContent = '0 Paare';
        } else {
            const totalDur = trainingPairs.reduce((s, p) => s + (p.duration_s || 0), 0);
            listEl.innerHTML = trainingPairs.map(p => {
                const t     = (p.text || '');
                const short = t.length > 38 ? t.slice(0, 36) + '…' : t;
                const esc   = escapeHtml(short);   // displayed (HTML text content)
                const safe  = escapeHtml(t);       // title attribute (quotes escaped too)
                const dur   = (p.duration_s || 0).toFixed(1);
                return `<div style="display:flex;align-items:center;gap:4px;padding:4px 6px;border-bottom:1px solid #2a2a2a;">
                    <span style="color:#555;font-size:10px;min-width:28px;">${p.id}</span>
                    <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;color:#ccc;" title="${safe}">${esc}</span>
                    <span style="color:#888;font-size:10px;min-width:30px;">${dur}s</span>
                    <button class="sp-play" data-id="${p.id}" style="background:#333;color:#aaa;border:1px solid #555;border-radius:4px;padding:2px 6px;cursor:pointer;font-size:11px;">▶</button>
                    <button class="sp-del"  data-id="${p.id}" style="background:#333;color:#c0392b;border:1px solid #555;border-radius:4px;padding:2px 6px;cursor:pointer;font-size:11px;">✕</button>
                </div>`;
            }).join('');
            if (countEl) countEl.textContent = `${trainingPairs.length} Paar${trainingPairs.length !== 1 ? 'e' : ''} · ${totalDur.toFixed(1)}s gesamt`;
        }
        listEl.onclick = async (e) => {
            const playBtn = e.target.closest('.sp-play');
            const delBtn  = e.target.closest('.sp-del');
            if (playBtn) {
                const id = playBtn.dataset.id;
                if (paaresCurrentAudio) {
                    paaresCurrentAudio.pause();
                    paaresCurrentAudio.src = '';
                    const prevBtn = listEl.querySelector(`.sp-play[data-id="${paaresCurrentPlayId}"]`);
                    if (prevBtn) prevBtn.textContent = '▶';
                    const wasId = paaresCurrentPlayId;
                    paaresCurrentAudio  = null;
                    paaresCurrentPlayId = null;
                    if (wasId === id) return;
                }
                // Fetch the WAV via authFetch (so the API key is sent), then play it
                // from an object URL — `new Audio(url)` cannot carry the auth header.
                let audio;
                try {
                    const res = await authFetch(`${getHttpBase()}/training/audio/${id}`);
                    if (!res.ok) throw new Error(await res.text());
                    const blobUrl = URL.createObjectURL(await res.blob());
                    audio = new Audio(blobUrl);
                    audio.addEventListener('ended', () => URL.revokeObjectURL(blobUrl), { once: true });
                } catch (err) {
                    setPaareStatus('Wiedergabe: ' + err.message, true);
                    return;
                }
                paaresCurrentAudio  = audio;
                paaresCurrentPlayId = id;
                playBtn.textContent = '⏹';
                audio.play().catch(err => {
                    paaresCurrentAudio  = null;
                    paaresCurrentPlayId = null;
                    playBtn.textContent = '▶';
                    setPaareStatus('Wiedergabe: ' + err.message, true);
                });
                audio.addEventListener('ended', () => {
                    paaresCurrentAudio  = null;
                    paaresCurrentPlayId = null;
                    playBtn.textContent = '▶';
                });
            }
            if (delBtn) {
                try {
                    const res = await authFetch(`${getHttpBase()}/training/pair/${delBtn.dataset.id}`, { method: 'DELETE' });
                    if (!res.ok) throw new Error(await res.text());
                    setPaareStatus(`Paar ${delBtn.dataset.id} gelöscht`, false);
                    await loadTrainingPairs();
                } catch (err) {
                    setPaareStatus('Fehler: ' + err.message, true);
                }
            }
        };
    }

    configPanel.querySelector('#schmidi-training-run').addEventListener('click', runLoraTraining);
    configPanel.querySelector('#schmidi-lora-reload').addEventListener('click', reloadLora);
    configPanel.querySelector('#schmidi-lora-enabled').addEventListener('change', toggleLora);

    // Checkbox: checked → load LoRA (POST /lora/reload with the adapter dir);
    // unchecked → unload it (DELETE /lora). The dir comes from GET /config's
    // `lora_dir`, so re-enabling works even after the active path was cleared.
    async function toggleLora(ev) {
        const box = ev.target;
        if (box.checked) {
            const dir = box.dataset.loraDir || '';
            setPaareStatus('Lade LoRA…', false);
            try {
                const res  = await authFetch(`${getHttpBase()}/lora/reload`, {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    dir ? JSON.stringify({ path: dir }) : undefined,
                });
                // Error bodies may be plain text (e.g. schmidiscribe's 422), so
                // parse defensively instead of assuming JSON.
                const raw = await res.text();
                let data = {};
                try { data = JSON.parse(raw); } catch (_) {}
                if (!res.ok) throw new Error(data.error || raw || res.status);
                if (data.action === 'applied') {
                    setPaareStatus('LoRA aktiv ✓', false);
                } else {
                    // No adapter files at the path — server reverted to base model.
                    box.checked = false;
                    setPaareStatus('Kein Adapter gefunden — LoRA nicht aktiv', true);
                }
            } catch (e) {
                box.checked = false;
                setPaareStatus('Fehler: ' + e.message, true);
            }
        } else {
            setPaareStatus('Entlade LoRA…', false);
            try {
                const res = await authFetch(`${getHttpBase()}/lora`, { method: 'DELETE' });
                if (!res.ok) throw new Error(await res.text());
                setPaareStatus('LoRA entladen (Basismodell)', false);
            } catch (e) {
                box.checked = true;
                setPaareStatus('Fehler: ' + e.message, true);
            }
        }
    }

    async function runLoraTraining() {
        setPaareStatus('Starte Training…', false);
        try {
            const res = await authFetch(`${getHttpBase()}/training/run`, { method: 'POST' });
            if (res.status === 409) { setPaareStatus('Training läuft bereits', false); return; }
            if (!res.ok) throw new Error(await res.text());
            setPaareStatus('Training gestartet', false);
            const logEl = configPanel.querySelector('#schmidi-training-log');
            if (logEl) logEl.style.display = 'block';
            if (trainingStatusPoll) clearInterval(trainingStatusPoll);
            trainingStatusPoll = setInterval(pollTrainingStatus, 2000);
        } catch (e) {
            setPaareStatus('Fehler: ' + e.message, true);
        }
    }

    async function reloadLora() {
        setPaareStatus('Lade LoRA neu…', false);
        try {
            const res = await authFetch(`${getHttpBase()}/lora/reload`, { method: 'POST' });
            const raw = await res.text();
            let data = {};
            try { data = JSON.parse(raw); } catch (_) {}
            if (!res.ok) throw new Error(data.error || raw || res.status);
            setPaareStatus(data.action === 'applied' ? 'LoRA geladen ✓' : 'LoRA entfernt (kein Adapter)', false);
        } catch (e) {
            setPaareStatus('Fehler: ' + e.message, true);
        }
    }

    async function pollTrainingStatus() {
        try {
            const res  = await authFetch(`${getHttpBase()}/training/status`);
            if (!res.ok) return;
            const data = await res.json();
            const statusMap = { idle: 'Bereit', running: 'Training läuft…', done: 'Training abgeschlossen ✓', error: 'Training fehlgeschlagen' };
            setPaareStatus(statusMap[data.status] || data.status, data.status === 'error');
            const logEl = configPanel.querySelector('#schmidi-training-log');
            if (logEl && data.log && data.log.length > 0) {
                logEl.textContent = data.log.join('\n');
                logEl.scrollTop   = logEl.scrollHeight;
            }
            if (data.status === 'done' || data.status === 'error') {
                clearInterval(trainingStatusPoll);
                trainingStatusPoll = null;
            }
        } catch (_) {}
    }

    // ---- Diktate tab (dictation review → training pairs) ----

    let reviewItems         = [];
    let reviewSelectedId    = null;
    let reviewCurrentAudio  = null;
    let reviewCurrentPlayId = null;

    function setDiktStatus(msg, isError) {
        const el = configPanel.querySelector('#schmidi-dikt-status');
        if (el) { el.textContent = msg; el.style.color = isError ? '#e74c3c' : '#aaa'; }
    }

    function updateLastDictationUi() {
        const saveBtn = configPanel.querySelector('#schmidi-dikt-save-last');
        const info    = configPanel.querySelector('#schmidi-dikt-last-info');
        if (!saveBtn || !info) return;
        if (!lastDictation || lastDictation.pcm.length === 0) {
            saveBtn.disabled = true;
            info.textContent = 'Kein Diktat in dieser Sitzung';
        } else {
            saveBtn.disabled = !!lastDictation.saved;
            const dur = (lastDictation.pcm.length / SAMPLE_RATE).toFixed(1);
            const t   = lastDictation.text.length > 60
                ? lastDictation.text.slice(0, 58) + '…' : lastDictation.text;
            info.textContent = (lastDictation.saved ? 'Gespeichert ✓ — ' : '') + `${dur}s: ${t}`;
        }
    }

    async function loadReviews() {
        updateLastDictationUi();
        setDiktStatus('Lade…', false);
        try {
            const res = await authFetch(`${getHttpBase()}/training/reviews`);
            if (!res.ok) throw new Error(res.status === 404
                ? 'Auf diesem Server nicht verfügbar' : 'GET /training/reviews: ' + res.status);
            const data = await res.json();
            reviewItems = data.reviews || [];
            renderReviewList();
            setDiktStatus('', false);
        } catch (e) {
            setDiktStatus('Fehler: ' + e.message, true);
        }
    }

    function renderReviewList() {
        if (reviewCurrentAudio) {
            reviewCurrentAudio.pause(); reviewCurrentAudio.src = '';
            reviewCurrentAudio = null; reviewCurrentPlayId = null;
        }
        const listEl = configPanel.querySelector('#schmidi-dikt-list');
        const detail = configPanel.querySelector('#schmidi-dikt-detail');
        if (!listEl) return;
        if (!reviewItems.some(r => r.id === reviewSelectedId)) reviewSelectedId = null;
        if (detail) detail.style.display = reviewSelectedId ? 'flex' : 'none';
        if (reviewItems.length === 0) {
            listEl.innerHTML = '<div style="padding:8px;color:#666;font-size:12px;">Keine gespeicherten Diktate</div>';
            return;
        }
        listEl.innerHTML = reviewItems.map(r => {
            const t     = r.text || '';
            const short = t.length > 34 ? t.slice(0, 32) + '…' : t;
            const sel   = r.id === reviewSelectedId;
            return `<div class="sp-dikt-row" data-id="${r.id}" style="display:flex;align-items:center;gap:4px;padding:4px 6px;border-bottom:1px solid #2a2a2a;cursor:pointer;${sel ? 'background:#26364a;' : ''}">
                <span style="color:#555;font-size:10px;min-width:28px;">${r.id}</span>
                <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;color:#ccc;" title="${escapeHtml(t)}">${escapeHtml(short)}</span>
                <span style="color:#888;font-size:10px;min-width:30px;">${(r.duration_s || 0).toFixed(1)}s</span>
                <button class="sp-dikt-play" data-id="${r.id}" style="background:#333;color:#aaa;border:1px solid #555;border-radius:4px;padding:2px 6px;cursor:pointer;font-size:11px;">▶</button>
                <button class="sp-dikt-del"  data-id="${r.id}" style="background:#333;color:#c0392b;border:1px solid #555;border-radius:4px;padding:2px 6px;cursor:pointer;font-size:11px;">✕</button>
            </div>`;
        }).join('');
        listEl.onclick = async (e) => {
            const playBtn = e.target.closest('.sp-dikt-play');
            const delBtn  = e.target.closest('.sp-dikt-del');
            const row     = e.target.closest('.sp-dikt-row');
            if (playBtn) { await playReviewAudio(playBtn); return; }
            if (delBtn)  { await deleteReview(delBtn.dataset.id); return; }
            if (row) selectReview(row.dataset.id);
        };
    }

    function selectReview(id) {
        reviewSelectedId = id;
        configPanel.querySelectorAll('.sp-dikt-row').forEach(r => {
            r.style.background = r.dataset.id === id ? '#26364a' : '';
        });
        const item = reviewItems.find(r => r.id === id);
        const ta   = configPanel.querySelector('#schmidi-dikt-text');
        if (ta) ta.value = (item && item.text) || '';
        const detail = configPanel.querySelector('#schmidi-dikt-detail');
        if (detail) detail.style.display = 'flex';
        setDiktStatus('', false);
    }

    async function playReviewAudio(playBtn) {
        const id     = playBtn.dataset.id;
        const listEl = configPanel.querySelector('#schmidi-dikt-list');
        if (reviewCurrentAudio) {
            reviewCurrentAudio.pause();
            reviewCurrentAudio.src = '';
            const prevBtn = listEl.querySelector(`.sp-dikt-play[data-id="${reviewCurrentPlayId}"]`);
            if (prevBtn) prevBtn.textContent = '▶';
            const wasId = reviewCurrentPlayId;
            reviewCurrentAudio  = null;
            reviewCurrentPlayId = null;
            if (wasId === id) return;
        }
        // authFetch + object URL so the API key is sent (same as Paare playback).
        let audio;
        try {
            const res = await authFetch(`${getHttpBase()}/training/review/audio/${id}`);
            if (!res.ok) throw new Error(await res.text());
            const blobUrl = URL.createObjectURL(await res.blob());
            audio = new Audio(blobUrl);
            audio.addEventListener('ended', () => URL.revokeObjectURL(blobUrl), { once: true });
        } catch (err) {
            setDiktStatus('Wiedergabe: ' + err.message, true);
            return;
        }
        reviewCurrentAudio  = audio;
        reviewCurrentPlayId = id;
        playBtn.textContent = '⏹';
        audio.play().catch(err => {
            reviewCurrentAudio  = null;
            reviewCurrentPlayId = null;
            playBtn.textContent = '▶';
            setDiktStatus('Wiedergabe: ' + err.message, true);
        });
        audio.addEventListener('ended', () => {
            reviewCurrentAudio  = null;
            reviewCurrentPlayId = null;
            playBtn.textContent = '▶';
        });
    }

    async function deleteReview(id) {
        try {
            const res = await authFetch(`${getHttpBase()}/training/review/${id}`, { method: 'DELETE' });
            if (!res.ok) throw new Error(await res.text());
            setDiktStatus(`Diktat ${id} gelöscht`, false);
            await loadReviews();
        } catch (e) {
            setDiktStatus('Fehler: ' + e.message, true);
        }
    }

    // Parse the server's 16-bit mono 16 kHz WAV into f32 PCM for WS replay.
    function parseWavPcm16(buf) {
        const dv = new DataView(buf);
        if (dv.byteLength < 12) throw new Error('WAV zu kurz');
        let off = 12; // skip RIFF/WAVE header
        while (off + 8 <= dv.byteLength) {
            const tag = String.fromCharCode(dv.getUint8(off), dv.getUint8(off + 1),
                                            dv.getUint8(off + 2), dv.getUint8(off + 3));
            const size = dv.getUint32(off + 4, true);
            if (tag === 'data') {
                const n = Math.min(Math.floor(size / 2), Math.floor((dv.byteLength - off - 8) / 2));
                const out = new Float32Array(n);
                for (let i = 0; i < n; i++) out[i] = dv.getInt16(off + 8 + i * 2, true) / 32768;
                return out;
            }
            off += 8 + size + (size % 2);
        }
        throw new Error('WAV: kein data-Chunk');
    }

    // Re-transcribe stored PCM by replaying it through a backend's normal WS
    // session — both servers speak the same protocol, so either model can be
    // chosen regardless of which backend is active. Resolves with the full
    // transcript (all finals + trailing partial).
    function transcribePcm(pcm, backend) {
        return new Promise((resolve, reject) => {
            let url = backendWsUrl(backend);
            const key = backendApiKey(backend);
            if (key) url += (url.includes('?') ? '&' : '?') + 'api_key=' + encodeURIComponent(key);
            const sock = new WebSocket(url);
            sock.binaryType = 'arraybuffer';
            let text        = '';
            let lastPartial = '';
            let settled     = false;
            // The server decodes at roughly real time — allow generous margin.
            const timeoutMs = Math.max(60000, (pcm.length / SAMPLE_RATE) * 4000 + 30000);
            const timer = setTimeout(() => fail('Zeitüberschreitung'), timeoutMs);
            function fail(msg) {
                if (!settled) { settled = true; clearTimeout(timer); reject(new Error(msg)); }
                try { sock.close(); } catch (_) {}
            }
            sock.onopen = () => {
                const CHUNK = SAMPLE_RATE; // 1 s per frame
                for (let i = 0; i < pcm.length; i += CHUNK) {
                    sock.send(pcm.subarray(i, Math.min(i + CHUNK, pcm.length)));
                }
                sock.send('stop');
            };
            sock.onmessage = (e) => {
                let m;
                try { m = JSON.parse(e.data); } catch (_) { return; }
                if (m.type === 'final')        { text += m.text || ''; lastPartial = ''; }
                else if (m.type === 'partial') { lastPartial = m.text || ''; }
                else if (m.type === 'error')   { fail(m.text || 'Serverfehler'); }
            };
            sock.onerror = () => fail('Verbindung fehlgeschlagen');
            sock.onclose = () => {
                clearTimeout(timer);
                if (!settled) { settled = true; resolve((text + lastPartial).trim()); }
            };
        });
    }

    async function transcribeReview(backend) {
        if (!reviewSelectedId) return;
        const label = BACKENDS[backend].label;
        setDiktStatus(`Transkribiere mit ${label}…`, false);
        try {
            const res = await authFetch(`${getHttpBase()}/training/review/audio/${reviewSelectedId}`);
            if (!res.ok) throw new Error(await res.text());
            const pcm = parseWavPcm16(await res.arrayBuffer());
            const out = await transcribePcm(pcm, backend);
            const ta  = configPanel.querySelector('#schmidi-dikt-text');
            if (ta) ta.value = out;
            setDiktStatus(`${label}-Transkript eingefügt — bitte korrigieren`, false);
        } catch (e) {
            setDiktStatus('Fehler: ' + e.message, true);
        }
    }

    async function acceptReview() {
        if (!reviewSelectedId) return;
        const ta   = configPanel.querySelector('#schmidi-dikt-text');
        const text = (ta ? ta.value : '').trim();
        if (!text) { setDiktStatus('Leerer Text', true); return; }
        setDiktStatus('Übernehme…', false);
        try {
            const res = await authFetch(`${getHttpBase()}/training/review/${reviewSelectedId}/accept`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ text }),
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            reviewSelectedId = null;
            await loadReviews();
            setDiktStatus(`Als Trainingspaar ${data.id} übernommen ✓`, false);
        } catch (e) {
            setDiktStatus('Fehler: ' + e.message, true);
        }
    }

    configPanel.querySelector('#schmidi-dikt-save-last').addEventListener('click', async () => {
        const ok = await saveDictationAsReview(undefined, (m) => setDiktStatus(m, false));
        if (ok) await loadReviews();
    });
    configPanel.querySelector('#schmidi-dikt-transcribe-voxtral').addEventListener('click', () => transcribeReview('voxtral'));
    configPanel.querySelector('#schmidi-dikt-transcribe-qwen').addEventListener('click', () => transcribeReview('qwen'));
    configPanel.querySelector('#schmidi-dikt-accept').addEventListener('click', acceptReview);
    configPanel.querySelector('#schmidi-dikt-text').addEventListener('keydown', (e) => e.stopPropagation());

    configPanel.querySelectorAll("input").forEach((input) => {
        input.addEventListener("keydown", (e) => {
            if (e.key === "Escape") closeConfig();
            e.stopPropagation();
        });
    });
    configPanel.querySelector("#schmidi-words").addEventListener("keydown", (e) => {
        e.stopPropagation();
    });
    configPanel.querySelector("#schmidi-words").addEventListener("input", () => {
        setWordsStatus("", false);
    });

    document.addEventListener("click", (e) => {
        if (!configPanel.contains(e.target) && e.target !== btn && !overlay.contains(e.target)) {
            closeConfig();
        }
    });

    // ---- Hotkey handling ----
    function matchesHotkey(e, hotkeyStr) {
        const parts     = hotkeyStr.toLowerCase().split("+");
        const key       = parts[parts.length - 1];
        const needCtrl  = parts.includes("ctrl");
        const needAlt   = parts.includes("alt");
        const needShift = parts.includes("shift");
        return (
            e.ctrlKey  === needCtrl  &&
            e.altKey   === needAlt   &&
            e.shiftKey === needShift &&
            e.key.toLowerCase() === key
        );
    }

    document.addEventListener(
        "keydown",
        (e) => {
            // Don't fire while typing in the config panel (e.g. a single-key hotkey).
            if (configPanel.contains(e.target)) return;
            if (!matchesHotkey(e, getHotkey())) return;
            e.preventDefault();
            e.stopPropagation();
            const active = document.activeElement;
            if (active && active !== document.body && active !== btn)
                targetEl = active;
            if (recording) stopRecording();
            else startRecording();
        },
        true,
    );

    btn.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        const active = document.activeElement;
        if (active && active !== btn && active !== document.body) {
            targetEl = active;
        }
    });

    btn.addEventListener("click", () => {
        closeConfig();
        if (recording) stopRecording();
        else startRecording();
    });

    btn.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        if (configPanel.style.display === "flex") closeConfig();
        else openConfig();
    });

    // ---- State helpers ----
    function setIdle() {
        btn.style.background = "#555";
        btn.style.animation  = "";
        btn.textContent      = "🎤";
    }
    function setListening() {
        btn.style.background = "#c0392b";
        btn.style.animation  = "schmidipulse 1s infinite";
        btn.textContent      = "⏹";
    }

    // ---- Toast ----
    function showToast(msg, duration = 3000) {
        const toast = document.createElement("div");
        Object.assign(toast.style, {
            position: "fixed", bottom: "90px", right: "90px",
            background: "rgba(30,30,30,0.95)", color: "#fff",
            padding: "10px 16px", borderRadius: "8px",
            zIndex: "2147483647", fontSize: "14px",
            maxWidth: "280px", pointerEvents: "none",
        });
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), duration);
    }

    // ---- Audio capture ----
    function startRecording() {
        accumulatedText     = "";
        currentPartial      = "";
        pendingText         = "";
        originalTranscribed = "";
        overlayUserEdited   = false;
        dictationPcmBuffers = [];
        navigator.mediaDevices
            .getUserMedia({ audio: true, video: false })
            .then((stream) => {
                micStream = stream;
                recording = true;
                reconnectAttempts = 0;
                connectWS(() => {
                    setupAudio(stream);
                    setListening();
                    updateOverlay();
                });
            })
            .catch((err) => {
                showToast("SCHMIDIspeech: Mikrofon nicht erlaubt — " + err.message);
            });
    }

    function setupAudio(stream) {
        audioContext = createCaptureContext();
        const source     = audioContext.createMediaStreamSource(stream);
        const bufSize    = 4096;
        scriptProcessor  = audioContext.createScriptProcessor(bufSize, 1, 1);
        source.connect(scriptProcessor);
        // Route through a silent gain node — ScriptProcessorNode needs a downstream
        // connection to fire onaudioprocess in some browsers, but connecting directly
        // to destination plays mic audio through the speakers and triggers AEC
        // self-cancellation (the echo canceller silences the speech it hears back).
        const silentOut = audioContext.createGain();
        silentOut.gain.value = 0;
        scriptProcessor.connect(silentOut);
        silentOut.connect(audioContext.destination);

        const ratio = audioContext.sampleRate / SAMPLE_RATE;
        scriptProcessor.onaudioprocess = (e) => {
            if (!recording || !ws || ws.readyState !== WebSocket.OPEN) return;
            const chunk = captureChunk(e.inputBuffer.getChannelData(0), ratio);
            ws.send(chunk.buffer);
            // Keep a copy of exactly what the server heard, so the finished
            // dictation can be saved as a training-pair candidate.
            dictationPcmBuffers.push(chunk);
        };
    }

    function stopRecording() {
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        recording = false;
        pendingOnReady = null;
        if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
        if (audioContext)    { audioContext.close(); audioContext = null; }
        if (micStream)       { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }

        if (ws && ws.readyState === WebSocket.OPEN) {
            // Send "stop" so the server flushes remaining text as a final before
            // closing. finalizeStop() runs on ws.onclose (server closed after the
            // flush) or after the 2s fallback timer.
            ws.send("stop");
            awaitingFinalFlush = true;
            stopFinalizeTimer = setTimeout(finalizeStop, 2000);
        } else {
            finalizeStop();
        }
    }

    // Commit trailing text and tear down the overlay after the server-side flush.
    function finalizeStop() {
        if (stopFinalizeTimer) { clearTimeout(stopFinalizeTimer); stopFinalizeTimer = null; }
        awaitingFinalFlush = false;
        if (ws) { ws.close(); ws = null; }

        if (isCommitMode()) {
            // Add trailing partial to pending text; snapshot for edit-log
            if (currentPartial) pendingText += currentPartial;
            currentPartial      = "";
            originalTranscribed = pendingText;
            snapshotDictation(pendingText);
            overlayUserEdited   = false;
            overlayTextDiv.textContent = pendingText;
            setIdle();
            updateOverlay();
        } else {
            // Live mode: inject any trailing partial, briefly keep overlay
            if (currentPartial) injectAtCursor(currentPartial);
            snapshotDictation(accumulatedText + currentPartial);
            accumulatedText = "";
            currentPartial  = "";
            setIdle();
            setTimeout(() => updateOverlay(), 2000);
        }
    }

    // Keep the finished dictation (audio + transcript) so it can be saved as a
    // training-pair candidate — via the overlay 💾 button or the Diktate tab.
    function snapshotDictation(text) {
        if (dictationPcmBuffers.length > 0) {
            lastDictation = {
                pcm:   concatPcm(dictationPcmBuffers),
                text:  (text || "").trim(),
                saved: false,
            };
        }
        dictationPcmBuffers = [];
    }

    // POST the last dictation (audio + transcript) to the server's review queue.
    // `text` overrides the snapshot transcript (e.g. the edited overlay text).
    async function saveDictationAsReview(text, statusFn) {
        const report = statusFn || showToast;
        if (!lastDictation || lastDictation.pcm.length === 0) {
            report("Kein Diktat vorhanden");
            return false;
        }
        const t = (text !== undefined ? text : lastDictation.text) || "";
        try {
            const res = await authFetch(
                `${getHttpBase()}/training/review?text=${encodeURIComponent(t)}`,
                { method: "POST", headers: { "Content-Type": "application/octet-stream" },
                  body: lastDictation.pcm.buffer });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            lastDictation.saved = true;
            report(`Diktat gespeichert ✓ (${data.id}, ${data.duration_s.toFixed(1)}s) — Review im Diktate-Tab`);
            return true;
        } catch (e) {
            report("Fehler beim Speichern: " + e.message);
            return false;
        }
    }

    // ---- WebSocket ----
    function connectWS(onReady) {
        // Remember the ready callback so it still runs if the *first* attempt
        // fails and a later retry is the one that actually connects.
        if (onReady) pendingOnReady = onReady;
        // Browsers cannot set custom headers on the WS upgrade, so the API key is
        // passed as a query param (?api_key=) instead of the X-Api-Key header.
        // hotwords/patient feed schmidiscribe's prompt biasing; voicetserver
        // ignores unknown query params, so they are always safe to send.
        let url = getServerUrl();
        const params = [];
        const key = apiKey();
        if (key) params.push("api_key=" + encodeURIComponent(key));
        const hw = hotwordsForUrl();
        if (hw) params.push("hotwords=" + encodeURIComponent(hw));
        const pat = getPatientName();
        if (pat) params.push("patient=" + encodeURIComponent(pat));
        if (params.length) url += (url.includes("?") ? "&" : "?") + params.join("&");
        ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            reconnectAttempts = 0;
            const cb = pendingOnReady;
            pendingOnReady = null;
            if (cb) cb();
        };

        ws.onmessage = (e) => {
            let msg;
            try { msg = JSON.parse(e.data); } catch (_) { return; }

            if (msg.type === "partial") {
                currentPartial = msg.text || "";
                updateOverlay();
            } else if (msg.type === "final") {
                // Fallback to currentPartial if server sends empty final
                const text = msg.text || currentPartial;
                if (text) {
                    if (isCommitMode()) {
                        pendingText += text;
                        if (!overlayUserEdited) {
                            overlayTextDiv.textContent = pendingText;
                        }
                    } else {
                        injectAtCursor(text);
                        accumulatedText += text;
                    }
                }
                currentPartial = "";
                updateOverlay();
            }
        };

        // A failed connection fires onerror *and* onclose. Let onclose own the
        // retry so a single failure doesn't schedule two reconnects (which would
        // leak a timer and open parallel sockets). onerror is just a no-op.
        ws.onerror = () => {};

        ws.onclose = () => {
            if (awaitingFinalFlush) {
                // Server closed after flushing the final in response to "stop".
                finalizeStop();
                return;
            }
            if (!recording) return;
            if (reconnectTimer) return;          // a retry is already pending
            if (reconnectAttempts < MAX_RECONNECT) {
                reconnectAttempts++;
                reconnectTimer = setTimeout(() => {
                    reconnectTimer = null;
                    connectWS(null);
                }, Math.pow(2, reconnectAttempts) * 500);
            } else {
                showToast("SCHMIDIspeech: Server nicht erreichbar — Aufnahme beendet");
                stopRecording();
            }
        };
    }

    // ---- Text injection — insert at current cursor position in targetEl ----
    function injectAtCursor(text) {
        const el = targetEl;
        if (!el || (!isEditable(el) && !isContentEditable(el))) {
            navigator.clipboard.writeText(text).catch(() => {});
            showToast("SCHMIDIspeech: Text in Zwischenablage kopiert");
            return;
        }
        if (isContentEditable(el)) {
            el.focus();
            document.execCommand("insertText", false, text);
        } else {
            const start  = el.selectionStart;
            const end    = el.selectionEnd;
            const before = el.value.substring(0, start);
            const after  = el.value.substring(end);
            el.value = before + text + after;
            const newPos = start + text.length;
            el.setSelectionRange(newPos, newPos);
            el.dispatchEvent(new InputEvent("input", {
                bubbles: true, cancelable: true, inputType: "insertText", data: text,
            }));
        }
    }

    function isEditable(el) {
        return el &&
            (el.tagName === "INPUT" || el.tagName === "TEXTAREA") &&
            !el.readOnly && !el.disabled;
    }
    function isContentEditable(el) {
        return el && el.isContentEditable;
    }
})();
