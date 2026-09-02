/**
 * The single source of time for the whiteboard.
 *
 * Plays one segment at a time from an <audio> element and reports progress
 * every animation frame from `audio.currentTime`. When a segment has no
 * audio (silent TTS engine) a virtual clock of `durationMs` is used instead
 * so subtitles, reveals and decorations still run in sync.
 */
export class AudioClock {
  constructor() {
    this.audio = new Audio();
    this.audio.preload = "auto";
    this.rate = 1.0;
    this.segment = null;      // { url, durationMs, onTick, onEnded }
    this._raf = null;
    this._virtualStart = 0;   // performance.now() at (virtual) play start
    this._virtualElapsed = 0; // ms accumulated before a pause
    this._paused = false;
    this.audio.addEventListener("ended", () => this._finish());
    this.audio.addEventListener("error", () => {
      if (this.segment && !this.segment.virtual) {
        // Fall back to a virtual clock so the session keeps flowing.
        this.segment.virtual = true;
        this._virtualStart = performance.now();
        this._virtualElapsed = 0;
      }
    });
  }

  get isPlaying() {
    return !!this.segment && !this._paused;
  }

  /** Current position in ms within the active segment. */
  get currentMs() {
    if (!this.segment) return 0;
    if (this.segment.virtual) {
      const running = this._paused ? 0 : (performance.now() - this._virtualStart) * this.rate;
      return Math.min(this.segment.durationMs, this._virtualElapsed + running);
    }
    return this.audio.currentTime * 1000;
  }

  get durationMs() {
    if (!this.segment) return 0;
    if (!this.segment.virtual && isFinite(this.audio.duration) && this.audio.duration > 0) {
      return this.audio.duration * 1000;
    }
    return this.segment.durationMs || 1;
  }

  play({ url, durationMs, onTick, onEnded }) {
    this.stop();
    this.segment = { url, durationMs: durationMs || 1000, onTick, onEnded, virtual: !url };
    this._paused = false;
    this._virtualElapsed = 0;
    this._virtualStart = performance.now();
    if (url) {
      this.audio.src = url;
      this.audio.playbackRate = this.rate;
      this.audio.play().catch(() => {
        // Autoplay blocked or decode failure: keep going virtually.
        this.segment.virtual = true;
        this._virtualStart = performance.now();
      });
    }
    this._tick();
  }

  pause() {
    if (!this.segment || this._paused) return;
    this._paused = true;
    if (this.segment.virtual) {
      this._virtualElapsed += (performance.now() - this._virtualStart) * this.rate;
    } else {
      this.audio.pause();
    }
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  resume() {
    if (!this.segment || !this._paused) return;
    this._paused = false;
    if (this.segment.virtual) {
      this._virtualStart = performance.now();
    } else {
      this.audio.play().catch(() => {});
    }
    this._tick();
  }

  setRate(rate) {
    if (this.segment?.virtual && !this._paused) {
      this._virtualElapsed += (performance.now() - this._virtualStart) * this.rate;
      this._virtualStart = performance.now();
    }
    this.rate = rate;
    this.audio.playbackRate = rate;
  }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this.segment = null;
    this._paused = false;
    try { this.audio.pause(); } catch {}
  }

  _tick() {
    const step = () => {
      if (!this.segment || this._paused) return;
      const ms = this.currentMs;
      const total = this.durationMs;
      const progress = Math.max(0, Math.min(1, ms / total));
      this.segment.onTick?.(progress, ms);
      if (this.segment.virtual && ms >= this.segment.durationMs) {
        this._finish();
        return;
      }
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }

  _finish() {
    const seg = this.segment;
    if (!seg) return;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this.segment = null;
    seg.onTick?.(1, seg.durationMs);
    seg.onEnded?.();
  }
}

/** Plays a one-off clip (interjection answers) without disturbing the main clock. */
export function playClip(url, durationMs) {
  return new Promise((resolve) => {
    if (!url) {
      setTimeout(resolve, Math.min(durationMs || 800, 1500));
      return;
    }
    const a = new Audio(url);
    a.addEventListener("ended", resolve, { once: true });
    a.addEventListener("error", resolve, { once: true });
    a.play().catch(resolve);
  });
}
