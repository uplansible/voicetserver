// ==UserScript==
// @name         SCHMIDIspeech
// @namespace    https://github.com/local/schmidispeech
// @version      0.1.0
// @description  Local GPU dictation — German medical (Voxtral Mini 4B Realtime)
// @match        *://*/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_notification
// ==/UserScript==

(function () {
    'use strict';

    // ---- Configuration ----
    // Server URL is stored in Violentmonkey storage so the user sets it once.
    // Default uses the Tailscale machine hostname pattern — edit to match yours.
    const DEFAULT_SERVER = 'ws://127.0.0.1:8765/asr'; // plain WS for dev
    // For production with Tailscale cert:
    // const DEFAULT_SERVER = 'wss://gpu-server.your-tailnet.ts.net:8765/asr';

    function getServerUrl() {
        return GM_getValue('server_url', DEFAULT_SERVER);
    }

    // ---- State ----
    let ws = null;
    let audioContext = null;
    let scriptProcessor = null;
    let micStream = null;
    let recording = false;
    let reconnectAttempts = 0;
    const MAX_RECONNECT = 3;
    const SAMPLE_RATE = 16000; // target rate for voicetserver

    // ---- UI ----
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
    btn.title = 'SCHMIDIspeech — click to dictate';
    document.body.appendChild(btn);

    // Tooltip for partial results
    const tooltip = document.createElement('div');
    Object.assign(tooltip.style, {
        position: 'fixed',
        bottom: '90px',
        right: '24px',
        maxWidth: '320px',
        background: 'rgba(30,30,30,0.92)',
        color: '#ddd',
        padding: '8px 12px',
        borderRadius: '8px',
        fontSize: '13px',
        lineHeight: '1.4',
        zIndex: '2147483647',
        display: 'none',
        wordBreak: 'break-word',
        pointerEvents: 'none',
    });
    document.body.appendChild(tooltip);

    // Dragging support
    let dragOffsetX = 0, dragOffsetY = 0, isDragging = false;
    btn.addEventListener('mousedown', (e) => {
        dragOffsetX = e.clientX - btn.getBoundingClientRect().left;
        dragOffsetY = e.clientY - btn.getBoundingClientRect().top;
        isDragging = false;
    });
    document.addEventListener('mousemove', (e) => {
        if (!e.buttons) return;
        isDragging = true;
        const x = e.clientX - dragOffsetX;
        const y = e.clientY - dragOffsetY;
        btn.style.left = x + 'px';
        btn.style.right = 'auto';
        btn.style.bottom = 'auto';
        btn.style.top = y + 'px';
        tooltip.style.left = x + 'px';
        tooltip.style.right = 'auto';
        tooltip.style.top = (y - 70) + 'px';
        tooltip.style.bottom = 'auto';
    });

    btn.addEventListener('click', () => {
        if (isDragging) return;
        if (recording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    // ---- State helpers ----
    function setIdle() {
        btn.style.background = '#555';
        btn.textContent = '🎤';
        tooltip.style.display = 'none';
    }
    function setListening() {
        btn.style.background = '#c0392b';
        btn.style.animation = 'schmidipulse 1s infinite';
        btn.textContent = '⏹';
    }
    function setProcessing() {
        btn.style.background = '#e67e22';
        btn.textContent = '⏳';
    }

    // Inject pulse animation
    const style = document.createElement('style');
    style.textContent = `@keyframes schmidipulse {
        0%,100% { box-shadow: 0 0 0 0 rgba(192,57,43,0.5); }
        50% { box-shadow: 0 0 0 10px rgba(192,57,43,0); }
    }`;
    document.head.appendChild(style);

    // ---- Toast notifications ----
    function showToast(msg, duration = 3000) {
        const toast = document.createElement('div');
        Object.assign(toast.style, {
            position: 'fixed',
            bottom: '90px',
            right: '90px',
            background: 'rgba(30,30,30,0.95)',
            color: '#fff',
            padding: '10px 16px',
            borderRadius: '8px',
            zIndex: '2147483647',
            fontSize: '14px',
            maxWidth: '280px',
            pointerEvents: 'none',
        });
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), duration);
    }

    // ---- Audio capture ----
    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then((stream) => {
                micStream = stream;
                recording = true;
                reconnectAttempts = 0;
                connectWS(() => {
                    setupAudio(stream);
                    setListening();
                });
            })
            .catch((err) => {
                showToast('SCHMIDIspeech: Mikrofon nicht erlaubt — ' + err.message);
            });
    }

    function setupAudio(stream) {
        // Use AudioContext at native rate, downsample to 16kHz manually
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        const nativeRate = audioContext.sampleRate;

        // ScriptProcessorNode (deprecated but universally available)
        // Process 4096 frames at a time
        const bufSize = 4096;
        scriptProcessor = audioContext.createScriptProcessor(bufSize, 1, 1);
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        let resampleBuf = [];
        const ratio = nativeRate / SAMPLE_RATE;

        scriptProcessor.onaudioprocess = (e) => {
            if (!recording || !ws || ws.readyState !== WebSocket.OPEN) return;

            const input = e.inputBuffer.getChannelData(0);

            // Simple nearest-neighbour downsample to 16kHz
            const outLen = Math.floor(input.length / ratio);
            const downsampled = new Float32Array(outLen);
            for (let i = 0; i < outLen; i++) {
                downsampled[i] = input[Math.floor(i * ratio)];
            }

            // Send raw f32 LE binary
            const buf = downsampled.buffer;
            ws.send(buf);
        };
    }

    function stopRecording() {
        recording = false;
        if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
        if (audioContext) { audioContext.close(); audioContext = null; }
        if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
        setIdle();
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
                tooltip.textContent = msg.text || '';
                tooltip.style.display = msg.text ? 'block' : 'none';
            } else if (msg.type === 'final') {
                tooltip.style.display = 'none';
                if (msg.text) {
                    injectText(msg.text);
                }
            }
        };

        ws.onerror = () => {
            if (reconnectAttempts < MAX_RECONNECT) {
                reconnectAttempts++;
                const delay = Math.pow(2, reconnectAttempts) * 500;
                setTimeout(() => connectWS(null), delay);
            } else {
                showToast('SCHMIDIspeech: Server nicht erreichbar — ' + url);
                stopRecording();
            }
        };

        ws.onclose = () => {
            if (recording && reconnectAttempts < MAX_RECONNECT) {
                reconnectAttempts++;
                const delay = Math.pow(2, reconnectAttempts) * 500;
                setTimeout(() => connectWS(null), delay);
            }
        };
    }

    // ---- Text injection ----
    function injectText(text) {
        const el = document.activeElement;

        if (!el || (!isEditable(el) && !isContentEditable(el))) {
            // No focused field — copy to clipboard + toast
            navigator.clipboard.writeText(text).catch(() => {});
            showToast('SCHMIDIspeech: Text in Zwischenablage kopiert');
            return;
        }

        if (isContentEditable(el)) {
            // contenteditable: use execCommand for undo support
            document.execCommand('insertText', false, text);
        } else {
            // input / textarea: insert at cursor position
            const start = el.selectionStart;
            const end = el.selectionEnd;
            const before = el.value.substring(0, start);
            const after = el.value.substring(end);
            el.value = before + text + after;
            const newPos = start + text.length;
            el.setSelectionRange(newPos, newPos);
            // Dispatch input event so frameworks (React, Vue, etc.) pick up the change
            el.dispatchEvent(new InputEvent('input', { bubbles: true, cancelable: true, inputType: 'insertText', data: text }));
        }
    }

    function isEditable(el) {
        return (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') && !el.readOnly && !el.disabled;
    }

    function isContentEditable(el) {
        return el.isContentEditable;
    }

})();
