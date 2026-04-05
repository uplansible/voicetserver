// ==UserScript==
// @name         SCHMIDIspeech
// @namespace    https://github.com/local/schmidispeech
// @version      0.1.5
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
        minWidth: '280px',
        boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
        fontSize: '13px',
    });
    configPanel.innerHTML = `
        <div style="font-weight:bold;margin-bottom:4px;">SCHMIDIspeech — Einstellungen</div>
        <div style="color:#aaa;font-size:11px;">Server URL</div>
        <input id="schmidi-url-input" type="text" style="
            background:#2a2a2a; color:#ddd; border:1px solid #555;
            border-radius:6px; padding:6px 8px; font-size:13px; width:100%;
            box-sizing:border-box; outline:none;
        "/>
        <div style="color:#aaa;font-size:11px;">Tastenkürzel (z.B. ctrl+shift+d, alt+s)</div>
        <input id="schmidi-hotkey-input" type="text" style="
            background:#2a2a2a; color:#ddd; border:1px solid #555;
            border-radius:6px; padding:6px 8px; font-size:13px; width:100%;
            box-sizing:border-box; outline:none;
        "/>
        <div style="display:flex;gap:8px;justify-content:flex-end;">
            <button id="schmidi-cancel" style="
                background:#333;color:#aaa;border:1px solid #555;
                border-radius:6px;padding:4px 12px;cursor:pointer;font-size:12px;">Abbrechen</button>
            <button id="schmidi-save" style="
                background:#2980b9;color:#fff;border:none;
                border-radius:6px;padding:4px 12px;cursor:pointer;font-size:12px;">Speichern</button>
        </div>
    `;
    document.body.appendChild(configPanel);

    function openConfig() {
        configPanel.querySelector('#schmidi-url-input').value = getServerUrl();
        configPanel.querySelector('#schmidi-hotkey-input').value = getHotkey();
        configPanel.style.display = 'flex';
        configPanel.querySelector('#schmidi-url-input').focus();
        configPanel.querySelector('#schmidi-url-input').select();
    }
    function closeConfig() { configPanel.style.display = 'none'; }

    configPanel.querySelector('#schmidi-cancel').addEventListener('click', closeConfig);
    configPanel.querySelector('#schmidi-save').addEventListener('click', () => {
        const url = configPanel.querySelector('#schmidi-url-input').value.trim();
        const hk  = configPanel.querySelector('#schmidi-hotkey-input').value.trim();
        if (url) setServerUrl(url);
        if (hk)  setHotkey(hk);
        btn.title = `SCHMIDIspeech — ${getHotkey()} zum Diktieren, Rechtsklick konfigurieren`;
        showToast('Einstellungen gespeichert');
        closeConfig();
    });
    configPanel.querySelectorAll('input').forEach((input) => {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') configPanel.querySelector('#schmidi-save').click();
            if (e.key === 'Escape') closeConfig();
            e.stopPropagation();
        });
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
