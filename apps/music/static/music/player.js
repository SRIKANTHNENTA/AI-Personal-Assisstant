(function () {
    const app = document.getElementById('musicApp');
    if (!app) {
        return;
    }

    const stateUrl = app.dataset.stateUrl;
    const actionUrl = app.dataset.actionUrl;

    const moodBadge = document.getElementById('moodBadge');
    const nowPlayingMood = document.getElementById('nowPlayingMood');
    const nowPlayingTitle = document.getElementById('nowPlayingTitle');
    const nowPlayingArtist = document.getElementById('nowPlayingArtist');
    const albumRecommendations = document.getElementById('albumRecommendations');
    const albumMoodBadge = document.getElementById('albumMoodBadge');

    const audio = document.getElementById('musicAudio');
    const seekBar = document.getElementById('seekBar');
    const volumeSlider = document.getElementById('volumeSlider');
    const currentTimeEl = document.getElementById('currentTime');
    const durationTimeEl = document.getElementById('durationTime');

    const prevBtn = document.getElementById('prevBtn');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const nextBtn = document.getElementById('nextBtn');

    const visualizerBars = Array.from(document.querySelectorAll('#audioVisualizer span'));
    const csrfInput = document.querySelector('#musicCsrfForm input[name="csrfmiddlewaretoken"]');

    let playerState = window.INITIAL_PLAYER_STATE || {};
    let ws = null;
    let audioContext = null;
    let analyser = null;
    let sourceNode = null;
    let visualizerFrame = null;
    let isSeeking = false;

    function getCsrfToken() {
        if (csrfInput && csrfInput.value) {
            return csrfInput.value;
        }
        const value = document.cookie.split('; ').find((row) => row.startsWith('csrftoken='));
        return value ? decodeURIComponent(value.split('=')[1]) : '';
    }

    function formatTime(seconds) {
        const total = Math.max(0, Math.floor(Number(seconds) || 0));
        const mins = Math.floor(total / 60);
        const secs = total % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function currentTrackUrl(track) {
        if (!track) {
            return '';
        }
        return track.stream_url || track.preview_url || '';
    }

    function ensureAudioPipeline() {
        if (audioContext) {
            return;
        }

        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) {
            return;
        }

        audioContext = new AudioCtx();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 128;
        analyser.smoothingTimeConstant = 0.83;

        sourceNode = audioContext.createMediaElementSource(audio);
        sourceNode.connect(analyser);
        analyser.connect(audioContext.destination);
    }

    function renderVisualizerIdle() {
        visualizerBars.forEach((bar, index) => {
            const idleHeight = 8 + ((index % 4) * 3);
            bar.style.height = `${idleHeight}px`;
        });
    }

    function animateVisualizer() {
        if (!analyser) {
            renderVisualizerIdle();
            return;
        }

        const buffer = new Uint8Array(analyser.frequencyBinCount);
        const draw = () => {
            analyser.getByteFrequencyData(buffer);
            for (let i = 0; i < visualizerBars.length; i += 1) {
                const value = buffer[i % buffer.length] || 0;
                const mapped = 8 + Math.max(0, Math.round((value / 255) * 84));
                visualizerBars[i].style.height = `${mapped}px`;
            }
            visualizerFrame = requestAnimationFrame(draw);
        };

        if (visualizerFrame) {
            cancelAnimationFrame(visualizerFrame);
        }
        visualizerFrame = requestAnimationFrame(draw);
    }

    function stopVisualizer() {
        if (visualizerFrame) {
            cancelAnimationFrame(visualizerFrame);
            visualizerFrame = null;
        }
        renderVisualizerIdle();
    }

    function setPlayButton(isPlaying) {
        playPauseBtn.innerHTML = isPlaying
            ? '<i class="fas fa-pause"></i>'
            : '<i class="fas fa-play"></i>';
    }

    function renderAlbumRecommendations(mapBySection) {
        if (!albumRecommendations) {
            return;
        }

        const sections = ['global', 'bollywood', 'tollywood'];
        albumRecommendations.innerHTML = sections.map((section) => {
            const tracks = Array.isArray(mapBySection?.[section]) ? mapBySection[section].slice(0, 4) : [];
            const rows = tracks.map((track) => {
                const albumTitle = track.album_title || 'Untitled album';
                const songTitle = track.song_title || albumTitle;
                const artist = track.artist || 'Unknown Artist';
                const preview = track.preview_url || '';
                const external = track.external_url || '';
                const artwork = track.artwork || '';
                const category = track.category || section;
                const mood = track.mood || playerState.mood || 'neutral';

                return `
                    <article class="music-list-item">
                        <div class="music-main-row">
                            <div class="music-row-left">
                                <div class="album-art">
                                    ${artwork ? `<img src="${artwork}" alt="${albumTitle} artwork">` : '<div class="album-art-fallback"><i class="fas fa-compact-disc"></i></div>'}
                                </div>
                                <div class="album-card-body">
                                    <div class="album-title">${albumTitle}</div>
                                    <div class="track-name text-secondary">${songTitle}</div>
                                    <div class="track-artist text-secondary">${artist}</div>
                                </div>
                            </div>
                            <div class="music-row-actions">
                                ${preview ? `<button type="button" class="btn btn-sm btn-accent js-preview-album" data-preview-url="${preview}" data-title="${songTitle}" data-artist="${artist}" data-album="${albumTitle}">Play</button>` : ''}
                                ${external ? `<a href="${external}" target="_blank" rel="noopener noreferrer" class="btn btn-sm btn-outline-light">Source</a>` : ''}
                                <button type="button" class="btn btn-sm btn-outline-light js-toggle-details">Details</button>
                            </div>
                        </div>
                        <div class="music-details" hidden>
                            <div class="album-meta small text-secondary">${category} | Mood: ${String(mood).replace(/^./, (c) => c.toUpperCase())}</div>
                            <div class="small text-secondary">Album: ${albumTitle} | Song: ${songTitle}</div>
                            <div class="small text-secondary">Artist: ${artist} | Source: ${(track.source || 'curated')}</div>
                        </div>
                    </article>
                `;
            }).join('');

            return `
                <section class="album-section" data-section="${section}">
                    <div class="album-header">
                        <h5 class="mb-0">${section.charAt(0).toUpperCase() + section.slice(1)}</h5>
                        <span class="small text-secondary">Album previews</span>
                    </div>
                    <div class="music-list">
                        ${rows || '<div class="small text-secondary">No tracks available right now for this section.</div>'}
                    </div>
                </section>
            `;
        }).join('');
    }

    async function postAction(action, extraPayload = {}) {
        const response = await fetch(actionUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken(),
            },
            body: JSON.stringify({ action, ...extraPayload }),
        });
        const data = await response.json();
        if (data && data.state) {
            applyState(data.state, { fromRemote: true, syncAudio: action !== 'seek' });
        }
    }

    async function startPlaybackOptimistically() {
        const track = playerState.current_track;
        const trackUrl = currentTrackUrl(track);
        if (!trackUrl) {
            return false;
        }

        ensureAudioPipeline();

        if (audio.src !== trackUrl) {
            audio.src = trackUrl;
            audio.load();
        }

        try {
            if (audioContext && audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            await audio.play();
            animateVisualizer();
            setPlayButton(true);
            return true;
        } catch (error) {
            console.warn('Playback unavailable:', error);
            setPlayButton(false);
            return false;
        }
    }

    async function startTrackPlayback(track) {
        const trackUrl = currentTrackUrl(track);
        if (!trackUrl) {
            return false;
        }

        ensureAudioPipeline();

        if (audio.src !== trackUrl) {
            audio.src = trackUrl;
            audio.load();
        }

        try {
            if (audioContext && audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            await audio.play();
            animateVisualizer();
            setPlayButton(true);
            return true;
        } catch (error) {
            console.warn('Playback unavailable:', error);
            setPlayButton(false);
            return false;
        }
    }

    function syncAudioToState(state, forceSeek) {
        const track = state.current_track;
        const nextUrl = currentTrackUrl(track);

        if (nextUrl && audio.src !== nextUrl) {
            audio.src = nextUrl;
            audio.load();
        }

        if (forceSeek && Number.isFinite(Number(state.current_time_seconds))) {
            audio.currentTime = Number(state.current_time_seconds);
        }

        if (Number.isFinite(Number(state.volume))) {
            audio.volume = Number(state.volume);
            volumeSlider.value = String(Number(state.volume));
        }

        const shouldPlay = state.playback_state === 'playing';
        setPlayButton(shouldPlay);

        if (shouldPlay && nextUrl) {
            ensureAudioPipeline();
            if (audioContext && audioContext.state === 'suspended') {
                audioContext.resume().catch(() => {});
            }
            audio.play().then(() => {
                animateVisualizer();
            }).catch(() => {
                setPlayButton(false);
            });
        } else {
            audio.pause();
            stopVisualizer();
        }
    }

    function applyState(state, options = {}) {
        if (!state) {
            return;
        }

        const fromRemote = Boolean(options.fromRemote);
        const syncAudio = options.syncAudio !== false;
        playerState = state;

        moodBadge.textContent = (state.mood || 'Neutral').replace(/^./, (c) => c.toUpperCase());
        if (albumMoodBadge) {
            albumMoodBadge.textContent = `Mood: ${moodBadge.textContent}`;
        }
        nowPlayingMood.textContent = moodBadge.textContent;

        const track = state.current_track;
        nowPlayingTitle.textContent = track ? track.title : 'No track selected';
        nowPlayingArtist.textContent = track && track.artist ? track.artist : '';

        renderAlbumRecommendations(state.album_recommendations || state.industry_recommendations || {});

        currentTimeEl.textContent = formatTime(state.current_time_seconds);
        durationTimeEl.textContent = formatTime(track ? track.duration_seconds : 0);

        if (!isSeeking) {
            const duration = Math.max(1, Number(track ? track.duration_seconds : 0));
            const pct = (Number(state.current_time_seconds || 0) / duration) * 100;
            seekBar.value = String(Math.min(100, Math.max(0, pct)));
        }

        if (syncAudio) {
            syncAudioToState(state, fromRemote);
        }
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        ws = new WebSocket(`${protocol}://${window.location.host}/ws/music/`);

        ws.onopen = () => {};

        ws.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload.type === 'music_state' && payload.state) {
                    applyState(payload.state, { fromRemote: true, syncAudio: true });
                }
            } catch (_error) {
                console.warn('Live update parse error; using fallback polling.');
            }
        };

        ws.onclose = () => {
            setTimeout(connectWebSocket, 2500);
        };

        ws.onerror = () => {};
    }

    function bindEvents() {
        playPauseBtn.addEventListener('click', async () => {
            const playing = playerState.playback_state === 'playing';
            if (playing) {
                audio.pause();
                stopVisualizer();
                setPlayButton(false);
                await postAction('pause');
                return;
            }

            const started = await startPlaybackOptimistically();
            await postAction('play');
            if (!started) {
                await fallbackInitialState();
            }
        });

        prevBtn.addEventListener('click', async () => {
            await postAction('previous');
        });

        nextBtn.addEventListener('click', async () => {
            await postAction('next');
        });

        if (albumRecommendations) {
            albumRecommendations.addEventListener('click', async (event) => {
                const detailsBtn = event.target.closest('.js-toggle-details');
                if (detailsBtn) {
                    event.preventDefault();
                    const item = detailsBtn.closest('.music-list-item');
                    const details = item ? item.querySelector('.music-details') : null;
                    if (!details) {
                        return;
                    }
                    const isHidden = details.hasAttribute('hidden');
                    details.toggleAttribute('hidden', !isHidden);
                    detailsBtn.textContent = isHidden ? 'Hide Details' : 'Details';
                    return;
                }

                const button = event.target.closest('.js-preview-album');
                if (!button) {
                    return;
                }

                event.preventDefault();
                const previewUrl = button.dataset.previewUrl || '';
                const title = button.dataset.title || 'Preview Track';
                const artist = button.dataset.artist || 'Free Source';
                if (!previewUrl) {
                    return;
                }

                const previewTrack = {
                    stream_url: previewUrl,
                    preview_url: previewUrl,
                    title,
                    artist,
                    duration_seconds: 30,
                };

                const started = await startTrackPlayback(previewTrack);
                if (started) {
                    nowPlayingTitle.textContent = title;
                    nowPlayingArtist.textContent = artist;
                }
            });
        }

        volumeSlider.addEventListener('input', async (event) => {
            const value = Number(event.target.value || 0.8);
            audio.volume = value;
            await postAction('volume', { volume: value });
        });

        seekBar.addEventListener('mousedown', () => {
            isSeeking = true;
        });

        seekBar.addEventListener('mouseup', async () => {
            isSeeking = false;
            const track = playerState.current_track;
            if (!track || !Number(track.duration_seconds)) {
                return;
            }
            const nextSeconds = (Number(seekBar.value) / 100) * Number(track.duration_seconds);
            audio.currentTime = nextSeconds;
            currentTimeEl.textContent = formatTime(nextSeconds);
            await postAction('seek', { current_time_seconds: nextSeconds });
        });

        audio.addEventListener('timeupdate', () => {
            if (!isSeeking) {
                const duration = Math.max(1, audio.duration || playerState.current_track?.duration_seconds || 1);
                const pct = (audio.currentTime / duration) * 100;
                seekBar.value = String(Math.min(100, Math.max(0, pct)));
                currentTimeEl.textContent = formatTime(audio.currentTime);
            }
        });

        audio.addEventListener('error', () => {
            const mediaError = audio.error;
            const errorText = mediaError ? `code ${mediaError.code}` : 'unknown source error';
            console.warn(`Audio source failed to load (${errorText}).`);
        });

        audio.addEventListener('ended', async () => {
            await postAction('next');
            await postAction('play');
        });

        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible' && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'request_state' }));
            }
        });
    }

    async function fallbackInitialState() {
        try {
            const response = await fetch(stateUrl, { method: 'GET' });
            const payload = await response.json();
            if (payload && payload.state) {
                applyState(payload.state, { fromRemote: true, syncAudio: true });
            }
        } catch (_error) {
            liveStatus.textContent = 'State sync failed.';
        }
    }

    bindEvents();
    renderVisualizerIdle();
    applyState(playerState, { fromRemote: false, syncAudio: false });
    fallbackInitialState();
    connectWebSocket();
})();
