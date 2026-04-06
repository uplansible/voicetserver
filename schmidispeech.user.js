// ==UserScript==
// @name         SCHMIDIspeech
// @namespace    https://github.com/local/schmidispeech
// @version      0.1.6
// @description  Local GPU dictation — German medical (Voxtral Mini 4B Realtime)
// @match        *://*/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_notification
// ==/UserScript==

(function () {
    'use strict';

    // ---- Configuration ----
    const DEFAULT_SERVER = 'ws://127.0.0.1:8765/asr';
    const DEFAULT_HOTKEY = 'ctrl+shift+d';

    function getServerUrl() { return GM_getValue('server_url', DEFAULT_SERVER); }
    function setServerUrl(url) { GM_setValue('server_url', url); }
    function getHotkey() { return GM_getValue('hotkey', DEFAULT_HOTKEY); }
    function setHotkey(k) { GM_setValue('hotkey', k.toLowerCase().trim()); }

    // ---- State ----
    let ws = null;
    let audioContext = null;
    let scriptProcessor = null;
    let micStream = null;
    let recording = false;
    let reconnectAttempts = 0;
    const MAX_RECONNECT = 3;
    const SAMPLE_RATE = 16000;

    // Target element for text injection — captured when dictation starts
    let targetEl = null;
    let accumulatedText = '';  // displayed in overlay (finals seen so far)
    let currentPartial = '';   // current unfinalized partial (overlay only)

    // ---- Styles ----
    const style = document.createElement('style');
    style.textContent = `
        @keyframes schmidipulse {
            0%,100% { box-shadow: 0 0 0 0 rgba(192,57,43,0.5); }
            50%      { box-shadow: 0 0 0 10px rgba(192,57,43,0); }
        }
        #schmidi-overlay-partial { color: #aaa; }
    `;
    document.head.appendChild(style);

    // ---- Mic button ----
    const btn = document.createElement('div');
    btn.id = 'schmidispeech-btn';
    Object.assign(btn.style, {
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        width: '56px',
        height: '56px',
        borderRadius: '50%',
        background: '#555',
        cursor: 'pointer',
        zIndex: '2147483647',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
        userSelect: 'none',
        transition: 'background 0.2s',
        fontSize: '24px',
        color: '#fff',
    });
    btn.textContent = '🎤';
    btn.title = `SCHMIDIspeech — ${getHotkey()} zum Diktieren, Rechtsklick konfigurieren`;
    document.body.appendChild(btn);

    // ---- Transcript overlay ----
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
        position: 'fixed',
        bottom: '90px',
        right: '24px',
        maxWidth: '360px',
        minWidth: '180px',
        background: 'rgba(20,20,20,0.93)',
        color: '#eee',
        padding: '10px 14px',
        borderRadius: '10px',
        fontSize: '14px',
        lineHeight: '1.5',
        zIndex: '2147483647',
        display: 'none',
        wordBreak: 'break-word',
        pointerEvents: 'none',
        boxShadow: '0 2px 12px rgba(0,0,0,0.5)',
    });
    document.body.appendChild(overlay);

    function updateOverlay() {
        if (!recording && !accumulatedText && !currentPartial) {
            overlay.style.display = 'none';
            return;
        }
        overlay.style.display = 'block';
        overlay.innerHTML =
            escapeHtml(accumulatedText) +
            '<span id="schmidi-overlay-partial">' + escapeHtml(currentPartial) + '</span>';
    }

    function escapeHtml(s) {
        return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    // ---- Config panel ----
    const configPanel = document.createElement('div');
    Object.assign(configPanel.style, {
        position: 'fixed',
        bottom: '90px',
        right: '24px',
        background: '#1e1e1e',
        border: '1px solid #444',
        color: '#ddd',
        padding: '12px',
        borderRadius: '10px',
        zIndex: '2147483647',
        display: 'none',
        flexDirection: 'column',
        gap: '8px',
        minWidth: '300px',
        boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
        fontSize: '13px',
    });

    const INPUT_STYLE = 'background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:6px;padding:6px 8px;font-size:13px;width:100%;box-sizing:border-box;outline:none;';
    const LABEL_STYLE = 'color:#aaa;font-size:11px;';
    const BTN_CANCEL  = 'background:#333;color:#aaa;border:1px solid #555;border-radius:6px;padding:4px 12px;cursor:pointer;font-size:12px;';
    const BTN_PRIMARY = 'background:#2980b9;color:#fff;border:none;border-radius:6px;padding:4px 12px;cursor:pointer;font-size:12px;';

    configPanel.innerHTML = `
        <div style="font-weight:bold;margin-bottom:2px;">SCHMIDIspeech — Einstellungen</div>
        <div style="display:flex;gap:0;border-bottom:1px solid #444;margin-bottom:4px;">
            <button id="schmidi-tab-client"   style="background:none;border:none;border-bottom:2px solid #2980b9;color:#ddd;padding:4px 10px;cursor:pointer;font-size:12px;font-weight:bold;">Client</button>
            <button id="schmidi-tab-server"   style="background:none;border:none;border-bottom:2px solid transparent;color:#888;padding:4px 10px;cursor:pointer;font-size:12px;">Server</button>
            <button id="schmidi-tab-training" style="background:none;border:none;border-bottom:2px solid transparent;color:#888;padding:4px 10px;cursor:pointer;font-size:12px;">Training</button>
        </div>

        <!-- Client tab -->
        <div id="schmidi-pane-client" style="display:flex;flex-direction:column;gap:8px;">
            <div style="${LABEL_STYLE}">Server URL</div>
            <input id="schmidi-url-input" type="text" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE}">Tastenkürzel (z.B. ctrl+shift+d, alt+s)</div>
            <input id="schmidi-hotkey-input" type="text" style="${INPUT_STYLE}"/>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="schmidi-cancel" style="${BTN_CANCEL}">Abbrechen</button>
                <button id="schmidi-save-client" style="${BTN_PRIMARY}">Speichern</button>
            </div>
        </div>

        <!-- Server tab -->
        <div id="schmidi-pane-server" style="display:none;flex-direction:column;gap:6px;">
            <div style="${LABEL_STYLE}">Delay tokens (1–30, je 80ms)</div>
            <input id="schmidi-delay" type="number" min="1" max="30" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE}">Silence threshold (RMS)</div>
            <input id="schmidi-silence-threshold" type="number" step="0.001" min="0" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE}">Silence detection chunks (je 80ms)</div>
            <input id="schmidi-silence-flush" type="number" min="1" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE}">Min speech chunks (je 80ms)</div>
            <input id="schmidi-min-speech" type="number" min="1" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE}">RMS EMA alpha (0.0–1.0)</div>
            <input id="schmidi-rms-ema" type="number" step="0.01" min="0" max="1" style="${INPUT_STYLE}"/>
            <div style="${LABEL_STYLE};margin-top:4px;">Eigene Wörter (ein Wort pro Zeile)</div>
            <textarea id="schmidi-words" rows="5" style="${INPUT_STYLE}font-family:monospace;resize:vertical;"></textarea>
            <div id="schmidi-server-status" style="color:#aaa;font-size:11px;min-height:14px;"></div>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="schmidi-cancel2" style="${BTN_CANCEL}">Abbrechen</button>
                <button id="schmidi-save-words" style="${BTN_CANCEL}">Wörter speichern</button>
                <button id="schmidi-save-server" style="${BTN_PRIMARY}">Params speichern</button>
            </div>
        </div>

        <!-- Training tab -->
        <div id="schmidi-pane-training" style="display:none;flex-direction:column;gap:6px;max-width:320px;">
            <div style="${LABEL_STYLE}">Kalibrierungssatz lesen</div>
            <div id="schmidi-training-sentence" style="background:#2a2a2a;border:1px solid #555;border-radius:6px;padding:8px;font-size:13px;line-height:1.4;min-height:40px;color:#eee;">Lade...</div>
            <div style="display:flex;gap:6px;align-items:center;">
                <button id="schmidi-prev-sentence" style="${BTN_CANCEL}">◀</button>
                <span id="schmidi-sentence-index" style="${LABEL_STYLE}">0/0</span>
                <button id="schmidi-next-sentence" style="${BTN_CANCEL}">▶</button>
            </div>
            <div style="${LABEL_STYLE}">Transkript (editierbar vor dem Speichern)</div>
            <input id="schmidi-training-text" type="text" style="${INPUT_STYLE}" placeholder="Aufnahme starten, dann hier korrigieren..."/>
            <div style="display:flex;gap:6px;">
                <button id="schmidi-training-record" style="${BTN_PRIMARY};flex:1;">⏺ Aufnehmen</button>
                <button id="schmidi-training-save"   style="${BTN_CANCEL};flex:1;" disabled>💾 Speichern</button>
            </div>
            <div id="schmidi-training-count" style="${LABEL_STYLE}">— Paare gespeichert</div>
            <div style="border-top:1px solid #444;margin:4px 0;"></div>
            <button id="schmidi-training-run" style="${BTN_PRIMARY}">▶ LoRA trainieren</button>
            <div id="schmidi-training-status" style="${LABEL_STYLE};min-height:14px;"></div>
            <pre id="schmidi-training-log" style="background:#111;border:1px solid #333;border-radius:4px;padding:6px;font-size:10px;max-height:80px;overflow-y:auto;margin:0;display:none;color:#8f8;"></pre>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button id="schmidi-cancel3" style="${BTN_CANCEL}">Schließen</button>
            </div>
        </div>
    `;
    document.body.appendChild(configPanel);

    // ---- Tab switching ----
    function switchTab(tab) {
        const tabs = ['client', 'server', 'training'];
        tabs.forEach(t => {
            const pane = configPanel.querySelector(`#schmidi-pane-${t}`);
            const btn_ = configPanel.querySelector(`#schmidi-tab-${t}`);
            if (!pane || !btn_) return;
            const active = (t === tab);
            pane.style.display = active ? 'flex' : 'none';
            btn_.style.borderBottomColor = active ? '#2980b9' : 'transparent';
            btn_.style.color = active ? '#ddd' : '#888';
        });
    }
    configPanel.querySelector('#schmidi-tab-client').addEventListener('click', () => switchTab('client'));
    configPanel.querySelector('#schmidi-tab-server').addEventListener('click', () => {
        switchTab('server');
        loadServerConfig();
    });
    configPanel.querySelector('#schmidi-tab-training').addEventListener('click', () => {
        switchTab('training');
        loadTrainingSentences();
        refreshTrainingCount();
    });

    // ---- HTTP helpers ----
    function getHttpBase() {
        return getServerUrl()
            .replace(/^ws(s?):\/\//, 'http$1://')
            .replace(/\/asr$/, '');
    }

    function setServerStatus(msg, isError) {
        const el = configPanel.querySelector('#schmidi-server-status');
        if (el) { el.textContent = msg; el.style.color = isError ? '#e74c3c' : '#aaa'; }
    }

    async function loadServerConfig() {
        setServerStatus('Lade Server-Einstellungen…', false);
        try {
            const [cfgRes, wordsRes] = await Promise.all([
                fetch(`${getHttpBase()}/config`),
                fetch(`${getHttpBase()}/words`),
            ]);
            if (!cfgRes.ok) throw new Error('GET /config: ' + cfgRes.status);
            const cfg = await cfgRes.json();
            configPanel.querySelector('#schmidi-delay').value            = cfg.delay            ?? '';
            configPanel.querySelector('#schmidi-silence-threshold').value = cfg.silence_threshold ?? '';
            configPanel.querySelector('#schmidi-silence-flush').value    = cfg.silence_flush    ?? '';
            configPanel.querySelector('#schmidi-min-speech').value       = cfg.min_speech       ?? '';
            configPanel.querySelector('#schmidi-rms-ema').value          = cfg.rms_ema          ?? '';

            if (wordsRes.ok) {
                const w = await wordsRes.json();
                configPanel.querySelector('#schmidi-words').value = (w.words || []).join('\n');
            }
            setServerStatus('', false);
        } catch (e) {
            setServerStatus('Fehler: ' + e.message, true);
        }
    }

    async function saveServerParams() {
        const patch = {
            delay:             parseInt(configPanel.querySelector('#schmidi-delay').value) || undefined,
            silence_threshold: parseFloat(configPanel.querySelector('#schmidi-silence-threshold').value) || undefined,
            silence_flush:     parseInt(configPanel.querySelector('#schmidi-silence-flush').value) || undefined,
            min_speech:        parseInt(configPanel.querySelector('#schmidi-min-speech').value) || undefined,
            rms_ema:           parseFloat(configPanel.querySelector('#schmidi-rms-ema').value) || undefined,
        };
        // Remove undefined keys
        Object.keys(patch).forEach(k => patch[k] === undefined && delete patch[k]);
        setServerStatus('Speichere…', false);
        try {
            const res = await fetch(`${getHttpBase()}/config`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patch),
            });
            if (!res.ok) throw new Error(await res.text());
            setServerStatus('Gespeichert ✓', false);
        } catch (e) {
            setServerStatus('Fehler: ' + e.message, true);
        }
    }

    async function saveWords() {
        const lines = configPanel.querySelector('#schmidi-words').value
            .split('\n').map(s => s.trim()).filter(Boolean);
        setServerStatus('Speichere Wörter…', false);
        try {
            // Compute diff against current state: fetch current list, then add/remove
            const wordsRes = await fetch(`${getHttpBase()}/words`);
            const current = wordsRes.ok ? (await wordsRes.json()).words || [] : [];
            const newSet  = new Set(lines);
            const curSet  = new Set(current);
            const add     = lines.filter(w => !curSet.has(w));
            const remove  = current.filter(w => !newSet.has(w));
            const res = await fetch(`${getHttpBase()}/words`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ add, remove }),
            });
            if (!res.ok) throw new Error(await res.text());
            setServerStatus(`Wörter gespeichert ✓ (${lines.length} Einträge)`, false);
        } catch (e) {
            setServerStatus('Fehler: ' + e.message, true);
        }
    }

    // ---- Config panel actions ----

    function openConfig() {
        configPanel.querySelector('#schmidi-url-input').value = getServerUrl();
        configPanel.querySelector('#schmidi-hotkey-input').value = getHotkey();
        switchTab('client');
        configPanel.style.display = 'flex';
        configPanel.querySelector('#schmidi-url-input').focus();
        configPanel.querySelector('#schmidi-url-input').select();
    }
    function closeConfig() { configPanel.style.display = 'none'; }

    configPanel.querySelector('#schmidi-cancel').addEventListener('click', closeConfig);
    configPanel.querySelector('#schmidi-cancel2').addEventListener('click', closeConfig);

    configPanel.querySelector('#schmidi-save-client').addEventListener('click', () => {
        const url = configPanel.querySelector('#schmidi-url-input').value.trim();
        const hk  = configPanel.querySelector('#schmidi-hotkey-input').value.trim();
        if (url) setServerUrl(url);
        if (hk)  setHotkey(hk);
        btn.title = `SCHMIDIspeech — ${getHotkey()} zum Diktieren, Rechtsklick konfigurieren`;
        showToast('Einstellungen gespeichert');
        closeConfig();
    });

    configPanel.querySelector('#schmidi-save-server').addEventListener('click', saveServerParams);
    configPanel.querySelector('#schmidi-save-words').addEventListener('click', saveWords);
    configPanel.querySelector('#schmidi-cancel3').addEventListener('click', closeConfig);

    // ---- Training tab state ----
    let trainingSentences = [];
    let trainingCurrentIdx = 0;
    let trainingRecording = false;
    let trainingPcmBuffers = [];
    let trainingAudioCtx = null;
    let trainingScriptProc = null;
    let trainingMicStream = null;
    let trainingStatusPoll = null;

    function setTrainingStatus(msg, isError) {
        const el = configPanel.querySelector('#schmidi-training-status');
        if (el) { el.textContent = msg; el.style.color = isError ? '#e74c3c' : '#aaa'; }
    }

    async function loadTrainingSentences() {
        try {
            const res = await fetch(`${getHttpBase()}/training/sentences`);
            if (!res.ok) throw new Error('GET /training/sentences: ' + res.status);
            const data = await res.json();
            trainingSentences = data.sentences || [];
            trainingCurrentIdx = Math.min(trainingCurrentIdx, Math.max(0, trainingSentences.length - 1));
            renderCurrentSentence();
        } catch (e) {
            setTrainingStatus('Fehler: ' + e.message, true);
        }
    }

    function renderCurrentSentence() {
        const sentEl = configPanel.querySelector('#schmidi-training-sentence');
        const idxEl  = configPanel.querySelector('#schmidi-sentence-index');
        const txtEl  = configPanel.querySelector('#schmidi-training-text');
        if (!sentEl || !idxEl) return;
        const total = trainingSentences.length;
        if (total === 0) {
            sentEl.textContent = 'Keine Sätze verfügbar';
            idxEl.textContent = '0/0';
            return;
        }
        const sentence = trainingSentences[trainingCurrentIdx] || '';
        sentEl.textContent = sentence;
        idxEl.textContent = `${trainingCurrentIdx + 1}/${total}`;
        if (txtEl && !trainingRecording) txtEl.value = sentence;
    }

    async function refreshTrainingCount() {
        try {
            const res = await fetch(`${getHttpBase()}/training`);
            if (!res.ok) return;
            const data = await res.json();
            const el = configPanel.querySelector('#schmidi-training-count');
            if (el) el.textContent = `${data.count} Paar${data.count !== 1 ? 'e' : ''} gespeichert (${(data.duration_sec || 0).toFixed(1)}s)`;
        } catch (_) {}
    }

    configPanel.querySelector('#schmidi-prev-sentence').addEventListener('click', () => {
        if (trainingSentences.length === 0) return;
        trainingCurrentIdx = (trainingCurrentIdx - 1 + trainingSentences.length) % trainingSentences.length;
        renderCurrentSentence();
    });
    configPanel.querySelector('#schmidi-next-sentence').addEventListener('click', () => {
        if (trainingSentences.length === 0) return;
        trainingCurrentIdx = (trainingCurrentIdx + 1) % trainingSentences.length;
        renderCurrentSentence();
    });

    configPanel.querySelector('#schmidi-training-record').addEventListener('click', () => {
        if (trainingRecording) stopTrainingRecording();
        else startTrainingRecording();
    });

    configPanel.querySelector('#schmidi-training-save').addEventListener('click', saveTrainingPair);

    configPanel.querySelector('#schmidi-training-run').addEventListener('click', runLoraTraining);

    // Detect best MediaRecorder format: OGG Vorbis (Firefox) → raw PCM fallback (Chrome)
    function getTrainingMimeType() {
        if (window.MediaRecorder) {
            const candidates = ['audio/ogg;codecs=vorbis', 'audio/ogg', 'audio/webm;codecs=pcm'];
            for (const mime of candidates) {
                if (MediaRecorder.isTypeSupported(mime)) return mime;
            }
        }
        return null;  // fall back to raw PCM via AudioWorklet
    }

    function startTrainingRecording() {
        trainingPcmBuffers = [];
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then((stream) => {
                trainingMicStream = stream;
                trainingRecording = true;
                const mimeType = getTrainingMimeType();

                if (mimeType && window.MediaRecorder) {
                    // Use MediaRecorder (OGG Vorbis on Firefox — lower bandwidth, server decodes with symphonia)
                    const mr = new MediaRecorder(stream, { mimeType });
                    mr.ondataavailable = (e) => {
                        if (e.data && e.data.size > 0) trainingPcmBuffers.push(e.data);
                    };
                    mr.start(100);  // 100ms chunks
                    trainingScriptProc = mr;  // reuse field to hold MediaRecorder ref
                    setTrainingStatus(`Aufnahme (${mimeType})…`, false);
                } else {
                    // Fallback: raw f32 LE PCM via ScriptProcessor (Chrome)
                    trainingAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const source   = trainingAudioCtx.createMediaStreamSource(stream);
                    const ratio    = trainingAudioCtx.sampleRate / SAMPLE_RATE;
                    const proc     = trainingAudioCtx.createScriptProcessor(4096, 1, 1);
                    source.connect(proc);
                    proc.connect(trainingAudioCtx.destination);
                    proc.onaudioprocess = (e) => {
                        if (!trainingRecording) return;
                        const input = e.inputBuffer.getChannelData(0);
                        const outLen = Math.floor(input.length / ratio);
                        const buf = new Float32Array(outLen);
                        for (let i = 0; i < outLen; i++) buf[i] = input[Math.floor(i * ratio)];
                        trainingPcmBuffers.push(buf);
                    };
                    trainingScriptProc = proc;
                    setTrainingStatus('Aufnahme (raw PCM)…', false);
                }

                const recBtn = configPanel.querySelector('#schmidi-training-record');
                if (recBtn) { recBtn.textContent = '⏹ Stop'; recBtn.style.background = '#c0392b'; }
            })
            .catch((err) => setTrainingStatus('Mikrofon: ' + err.message, true));
    }

    function stopTrainingRecording() {
        trainingRecording = false;

        // Stop MediaRecorder or ScriptProcessor
        if (trainingScriptProc instanceof MediaRecorder) {
            trainingScriptProc.stop();
            // Wait briefly for final ondataavailable
            setTimeout(finalizeTrainingRecording, 200);
        } else {
            if (trainingScriptProc) { trainingScriptProc.disconnect(); }
            if (trainingAudioCtx)   { trainingAudioCtx.close(); trainingAudioCtx = null; }
            trainingScriptProc = null;
            finalizeTrainingRecording();
        }
        if (trainingMicStream) { trainingMicStream.getTracks().forEach(t => t.stop()); trainingMicStream = null; }
    }

    function finalizeTrainingRecording() {
        const recBtn  = configPanel.querySelector('#schmidi-training-record');
        const saveBtn = configPanel.querySelector('#schmidi-training-save');
        if (recBtn)  { recBtn.textContent = '⏺ Aufnehmen'; recBtn.style.background = '#2980b9'; }
        if (saveBtn) saveBtn.disabled = (trainingPcmBuffers.length === 0);
        const dur = trainingPcmBuffers.length > 0 ? '…' : '0';
        setTrainingStatus(`Aufnahme beendet (${trainingPcmBuffers.length} Blöcke)`, false);
    }

    async function saveTrainingPair() {
        if (trainingPcmBuffers.length === 0) return;
        const txtEl = configPanel.querySelector('#schmidi-training-text');
        const text  = (txtEl ? txtEl.value : '').trim();
        if (!text) { setTrainingStatus('Kein Transkript eingegeben', true); return; }

        // Combine buffers: either Blobs (MediaRecorder) or Float32Arrays (raw PCM)
        let body, contentType;
        if (trainingPcmBuffers[0] instanceof Blob) {
            // OGG Vorbis / WebM from MediaRecorder — server decodes with symphonia
            body        = new Blob(trainingPcmBuffers, { type: trainingPcmBuffers[0].type });
            contentType = trainingPcmBuffers[0].type || 'application/octet-stream';
        } else {
            // Raw f32 LE PCM buffers — server decodes with decode_pcm_f32
            const totalLen = trainingPcmBuffers.reduce((s, b) => s + b.length, 0);
            const combined = new Float32Array(totalLen);
            let offset = 0;
            for (const buf of trainingPcmBuffers) { combined.set(buf, offset); offset += buf.length; }
            body        = combined.buffer;
            contentType = 'application/octet-stream';
        }

        setTrainingStatus('Speichere…', false);
        try {
            const res = await fetch(`${getHttpBase()}/training/pair?text=${encodeURIComponent(text)}`, {
                method: 'POST',
                headers: { 'Content-Type': contentType },
                body,
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            setTrainingStatus(`Gespeichert ✓ (Paar ${data.id}, ${data.duration_s.toFixed(1)}s)`, false);
            trainingPcmBuffers = [];
            const saveBtn = configPanel.querySelector('#schmidi-training-save');
            if (saveBtn) saveBtn.disabled = true;
            refreshTrainingCount();
            // Advance to next sentence
            if (trainingSentences.length > 1) {
                trainingCurrentIdx = (trainingCurrentIdx + 1) % trainingSentences.length;
                renderCurrentSentence();
            }
        } catch (e) {
            setTrainingStatus('Fehler: ' + e.message, true);
        }
    }

    async function runLoraTraining() {
        setTrainingStatus('Starte Training…', false);
        try {
            const res = await fetch(`${getHttpBase()}/training/run`, { method: 'POST' });
            if (res.status === 409) { setTrainingStatus('Training läuft bereits', false); return; }
            if (!res.ok) throw new Error(await res.text());
            setTrainingStatus('Training gestartet', false);
            // Show log panel and start polling
            const logEl = configPanel.querySelector('#schmidi-training-log');
            if (logEl) logEl.style.display = 'block';
            if (trainingStatusPoll) clearInterval(trainingStatusPoll);
            trainingStatusPoll = setInterval(pollTrainingStatus, 2000);
        } catch (e) {
            setTrainingStatus('Fehler: ' + e.message, true);
        }
    }

    async function pollTrainingStatus() {
        try {
            const res = await fetch(`${getHttpBase()}/training/status`);
            if (!res.ok) return;
            const data = await res.json();
            const statusMap = { idle: 'Bereit', running: 'Training läuft…', done: 'Training abgeschlossen ✓', error: 'Training fehlgeschlagen' };
            setTrainingStatus(statusMap[data.status] || data.status, data.status === 'error');
            const logEl = configPanel.querySelector('#schmidi-training-log');
            if (logEl && data.log && data.log.length > 0) {
                logEl.textContent = data.log.join('\n');
                logEl.scrollTop = logEl.scrollHeight;
            }
            if (data.status === 'done' || data.status === 'error') {
                clearInterval(trainingStatusPoll);
                trainingStatusPoll = null;
            }
        } catch (_) {}
    }

    configPanel.querySelectorAll('input').forEach((input) => {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeConfig();
            e.stopPropagation();
        });
    });
    configPanel.querySelector('#schmidi-words').addEventListener('keydown', (e) => {
        e.stopPropagation();
    });
    document.addEventListener('click', (e) => {
        if (!configPanel.contains(e.target) && e.target !== btn) closeConfig();
    });

    // ---- Hotkey handling ----
    function matchesHotkey(e, hotkeyStr) {
        // hotkeyStr format: "ctrl+shift+d", "alt+s", "ctrl+alt+m", etc.
        const parts = hotkeyStr.toLowerCase().split('+');
        const key = parts[parts.length - 1];
        const needCtrl  = parts.includes('ctrl');
        const needAlt   = parts.includes('alt');
        const needShift = parts.includes('shift');
        return (
            e.ctrlKey  === needCtrl  &&
            e.altKey   === needAlt   &&
            e.shiftKey === needShift &&
            e.key.toLowerCase() === key
        );
    }

    // Capture phase so we fire before page handlers; stopPropagation prevents page from acting on it
    document.addEventListener('keydown', (e) => {
        if (!matchesHotkey(e, getHotkey())) return;
        e.preventDefault();
        e.stopPropagation();
        // Capture focused element now — user is still in the field when pressing the hotkey
        const active = document.activeElement;
        if (active && active !== document.body && active !== btn) targetEl = active;
        if (recording) stopRecording(); else startRecording();
    }, true);

    // Capture focused element on mousedown (before button steals focus) — fallback for mouse users
    btn.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        const active = document.activeElement;
        if (active && active !== btn && active !== document.body) {
            targetEl = active;
        }
    });

    btn.addEventListener('click', () => {
        closeConfig();
        if (recording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    btn.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        if (configPanel.style.display === 'flex') closeConfig();
        else openConfig();
    });

    // ---- State helpers ----
    function setIdle()       { btn.style.background = '#555'; btn.style.animation = ''; btn.textContent = '🎤'; }
    function setListening()  { btn.style.background = '#c0392b'; btn.style.animation = 'schmidipulse 1s infinite'; btn.textContent = '⏹'; }

    // ---- Toast ----
    function showToast(msg, duration = 3000) {
        const toast = document.createElement('div');
        Object.assign(toast.style, {
            position: 'fixed', bottom: '90px', right: '90px',
            background: 'rgba(30,30,30,0.95)', color: '#fff',
            padding: '10px 16px', borderRadius: '8px',
            zIndex: '2147483647', fontSize: '14px', maxWidth: '280px',
            pointerEvents: 'none',
        });
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), duration);
    }

    // ---- Audio capture ----
    function startRecording() {
        accumulatedText = '';
        currentPartial = '';
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
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
                showToast('SCHMIDIspeech: Mikrofon nicht erlaubt — ' + err.message);
            });
    }

    function setupAudio(stream) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        const nativeRate = audioContext.sampleRate;
        const bufSize = 4096;
        scriptProcessor = audioContext.createScriptProcessor(bufSize, 1, 1);
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        const ratio = nativeRate / SAMPLE_RATE;
        scriptProcessor.onaudioprocess = (e) => {
            if (!recording || !ws || ws.readyState !== WebSocket.OPEN) return;
            const input = e.inputBuffer.getChannelData(0);
            const outLen = Math.floor(input.length / ratio);
            const downsampled = new Float32Array(outLen);
            for (let i = 0; i < outLen; i++) downsampled[i] = input[Math.floor(i * ratio)];
            ws.send(downsampled.buffer);
        };
    }

    function stopRecording() {
        recording = false;
        if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
        if (audioContext)     { audioContext.close();         audioContext = null; }
        if (micStream)        { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
        if (ws && ws.readyState === WebSocket.OPEN) ws.close();

        // Insert any trailing partial that hadn't been finalized yet
        if (currentPartial) injectAtCursor(currentPartial);

        accumulatedText = '';
        currentPartial = '';
        setIdle();
        // Keep overlay visible briefly so user can see what was inserted
        setTimeout(() => updateOverlay(), 2000);
    }

    // ---- WebSocket ----
    function connectWS(onReady) {
        const url = getServerUrl();
        ws = new WebSocket(url);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            reconnectAttempts = 0;
            if (onReady) onReady();
        };

        ws.onmessage = (e) => {
            let msg;
            try { msg = JSON.parse(e.data); } catch (_) { return; }

            if (msg.type === 'partial') {
                currentPartial = msg.text || '';
                updateOverlay();
            } else if (msg.type === 'final') {
                // Fallback to currentPartial if server sends empty final (e.g. silence detected
                // before model committed tokens to text_buf)
                const text = msg.text || currentPartial;
                if (text) {
                    injectAtCursor(text);
                    accumulatedText += text;
                }
                currentPartial = '';
                updateOverlay();
            }
        };

        ws.onerror = () => {
            if (reconnectAttempts < MAX_RECONNECT) {
                reconnectAttempts++;
                setTimeout(() => connectWS(null), Math.pow(2, reconnectAttempts) * 500);
            } else {
                showToast('SCHMIDIspeech: Server nicht erreichbar — ' + url);
                stopRecording();
            }
        };

        ws.onclose = () => {
            if (recording && reconnectAttempts < MAX_RECONNECT) {
                reconnectAttempts++;
                setTimeout(() => connectWS(null), Math.pow(2, reconnectAttempts) * 500);
            }
        };
    }

    // ---- Text injection — insert at current cursor position in targetEl ----
    function injectAtCursor(text) {
        const el = targetEl;

        if (!el || (!isEditable(el) && !isContentEditable(el))) {
            navigator.clipboard.writeText(text).catch(() => {});
            showToast('SCHMIDIspeech: Text in Zwischenablage kopiert');
            return;
        }

        if (isContentEditable(el)) {
            el.focus();
            document.execCommand('insertText', false, text);
        } else {
            // input / textarea — insert at current cursor position
            const start = el.selectionStart;
            const end   = el.selectionEnd;
            const before = el.value.substring(0, start);
            const after  = el.value.substring(end);
            el.value = before + text + after;
            const newPos = start + text.length;
            el.setSelectionRange(newPos, newPos);
            el.dispatchEvent(new InputEvent('input', {
                bubbles: true, cancelable: true,
                inputType: 'insertText', data: text,
            }));
        }
    }

    function isEditable(el) {
        return el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') && !el.readOnly && !el.disabled;
    }
    function isContentEditable(el) {
        return el && el.isContentEditable;
    }

})();
