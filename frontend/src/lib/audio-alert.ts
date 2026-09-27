/**
 * SARTHI Vision - Automotive ADAS Audio Alert Engine
 * 
 * Multi-tiered zero-latency audio alert system:
 * 1. Web Audio API synthesized ADAS radar collision chime (0ms instantaneous auditory warning for imminent hazards)
 * 2. Backend OpenAI TTS-1 Onyx authoritative voice stream
 * 3. Instant zero-latency Web Speech API fallback (local browser neural speech synthesis)
 * 4. User-gesture audio unlocking and queue deduplication
 * 5. Instant stop/pause alignment: zero orphan speech on pause/seek/stop.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

let audioCtx: AudioContext | null = null;
let isUnlocked = false;

/**
 * Unlock Web Audio and SpeechSynthesis on first user interaction.
 * Crucial for bypassing browser autoplay restrictions.
 */
export function unlockAudio(): void {
  if (isUnlocked && audioCtx?.state === "running") return;

  try {
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;

    if (!audioCtx && AudioContextClass) {
      audioCtx = new AudioContextClass();
    }

    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume();
    }

    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.resume();
    }

    isUnlocked = true;
  } catch (err) {
    console.warn("AudioContext unlock error:", err);
  }
}

/**
 * Play authentic automotive ADAS forward collision warning chime.
 * Synthesizes a subtle, gentle dual-tone cue chime (880Hz -> 1320Hz)
 * carefully calibrated to a pleasant volume so spoken voice warnings
 * remain loud, clear, and prominent without being drowned out.
 */
export function playUrgentCollisionChime(): void {
  unlockAudio();
  if (!audioCtx) return;

  try {
    const now = audioCtx.currentTime;
    
    // First gentle chirp (whisper-soft cue peak: 0.02)
    const osc1 = audioCtx.createOscillator();
    const gain1 = audioCtx.createGain();
    osc1.type = "sine";
    osc1.frequency.setValueAtTime(880, now);
    osc1.frequency.exponentialRampToValueAtTime(1240, now + 0.07);
    
    gain1.gain.setValueAtTime(0.001, now);
    gain1.gain.linearRampToValueAtTime(0.02, now + 0.015);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.08);

    osc1.connect(gain1);
    gain1.connect(audioCtx.destination);
    osc1.start(now);
    osc1.stop(now + 0.09);

    // Second gentle chirp (whisper-soft cue peak: 0.024)
    const osc2 = audioCtx.createOscillator();
    const gain2 = audioCtx.createGain();
    osc2.type = "sine";
    osc2.frequency.setValueAtTime(988, now + 0.10);
    osc2.frequency.exponentialRampToValueAtTime(1380, now + 0.17);

    gain2.gain.setValueAtTime(0.001, now + 0.10);
    gain2.gain.linearRampToValueAtTime(0.024, now + 0.115);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.18);

    osc2.connect(gain2);
    gain2.connect(audioCtx.destination);
    osc2.start(now + 0.10);
    osc2.stop(now + 0.19);
  } catch (e) {
    console.warn("Collision chime synthesis error:", e);
  }
}

interface AlertItem {
  phrase: string;
  urgency: "imminent" | "high" | "moderate";
  timestamp: number;
}

class AlertQueueManager {
  private queue: AlertItem[] = [];
  private isPlaying = false;
  private isHalted = false;
  private sessionId = 0;
  private nextTimer: ReturnType<typeof setTimeout> | null = null;
  private recentPhrases: Map<string, number> = new Map();
  private onSpeakingChange?: (phrase: string | null) => void;
  private currentAudio: HTMLAudioElement | null = null;
  private currentAbortController: AbortController | null = null;
  private currentUtterance: SpeechSynthesisUtterance | null = null;

  public setSpeakingCallback(cb: (phrase: string | null) => void) {
    this.onSpeakingChange = cb;
  }

  public resume(): void {
    this.isHalted = false;
  }

  public enqueue(
    phrase: string,
    urgency: "imminent" | "high" | "moderate" = "high",
    hazardKey?: string
  ): void {
    if (this.isHalted) return;
    unlockAudio();
    const now = Date.now();
    const cleanPhrase = phrase.trim();
    if (!cleanPhrase) return;

    // Cooldown per hazard key or normalized phrase: 3.5s to prevent audio drift
    const key = hazardKey ? hazardKey.toLowerCase() : cleanPhrase.toLowerCase().slice(0, 32);
    const lastPlayed = this.recentPhrases.get(key);
    if (lastPlayed && now - lastPlayed < 3500) {
      return;
    }

    const item: AlertItem = { phrase: cleanPhrase, urgency, timestamp: now };

    if (urgency === "imminent") {
      playUrgentCollisionChime();
      // Imminent collision alerts immediately take priority
      this.queue = [item];
    } else {
      // ADAS queue depth cap: never allow backlog of voice alerts
      if (this.queue.length === 0) {
        this.queue.push(item);
      } else if (this.queue[0].urgency !== "imminent") {
        this.queue = [item];
      }
    }

    if (!this.isPlaying) {
      this.processNext();
    }
  }

  private async processNext(): Promise<void> {
    if (this.isHalted || this.queue.length === 0) {
      this.isPlaying = false;
      this.onSpeakingChange?.(null);
      return;
    }

    const currentSession = this.sessionId;
    this.isPlaying = true;
    const item = this.queue.shift()!;
    const key = item.phrase.toLowerCase().slice(0, 32);
    this.recentPhrases.set(key, Date.now());
    this.onSpeakingChange?.(item.phrase);

    // If an imminent warning chime was just played, let the gentle cue chime conclude
    // (200ms) so the spoken words are completely clear and unobstructed
    if (item.urgency === "imminent") {
      await new Promise((resolve) => setTimeout(resolve, 200));
      if (this.isHalted || this.sessionId !== currentSession) {
        this.isPlaying = false;
        this.onSpeakingChange?.(null);
        return;
      }
    }

    try {
      await this.speak(item.phrase, currentSession);
    } catch (err) {
      console.warn("Speech playback error:", err);
    } finally {
      if (!this.isHalted && this.sessionId === currentSession) {
        this.nextTimer = setTimeout(() => {
          if (!this.isHalted && this.sessionId === currentSession) {
            this.processNext();
          } else {
            this.isPlaying = false;
            this.onSpeakingChange?.(null);
          }
        }, 150);
      } else {
        this.isPlaying = false;
        this.onSpeakingChange?.(null);
      }
    }
  }

  private speak(phrase: string, expectedSession: number): Promise<void> {
    return new Promise((resolve) => {
      if (this.isHalted || this.sessionId !== expectedSession) {
        resolve();
        return;
      }

      let resolved = false;
      const done = () => {
        if (!resolved) {
          resolved = true;
          this.currentAudio = null;
          this.currentAbortController = null;
          resolve();
        }
      };

      const safetyTimeout = setTimeout(() => {
        done();
      }, 6000);

      // 1. Try Backend OpenAI TTS endpoint
      const controller = new AbortController();
      this.currentAbortController = controller;
      const fetchTimeout = setTimeout(() => {
        try {
          controller.abort();
        } catch {
          // ignore
        }
      }, 2000); // 2000ms healthy network window for TTS stream

      fetch(`${API_BASE}/api/voice-alert?text=${encodeURIComponent(phrase)}`, {
        signal: controller.signal,
      })
        .then(async (resp) => {
          clearTimeout(fetchTimeout);
          if (this.isHalted || this.sessionId !== expectedSession) {
            clearTimeout(safetyTimeout);
            done();
            return;
          }
          if (!resp.ok) throw new Error("TTS endpoint error: " + resp.status);
          const blob = await resp.blob();
          if (this.isHalted || this.sessionId !== expectedSession) {
            clearTimeout(safetyTimeout);
            done();
            return;
          }
          const url = URL.createObjectURL(blob);
          const audio = new Audio(url);
          audio.volume = 1.0;
          this.currentAudio = audio;

          audio.onended = () => {
            URL.revokeObjectURL(url);
            clearTimeout(safetyTimeout);
            done();
          };
          audio.onerror = () => {
            URL.revokeObjectURL(url);
            if (!this.isHalted && this.sessionId === expectedSession) {
              this.speakWebSpeech(phrase, expectedSession, () => {
                clearTimeout(safetyTimeout);
                done();
              });
            } else {
              clearTimeout(safetyTimeout);
              done();
            }
          };

          try {
            await audio.play();
          } catch (playErr) {
            console.warn("HTML5 audio.play() blocked or failed, falling back to Web Speech:", playErr);
            URL.revokeObjectURL(url);
            if (!this.isHalted && this.sessionId === expectedSession) {
              this.speakWebSpeech(phrase, expectedSession, () => {
                clearTimeout(safetyTimeout);
                done();
              });
            } else {
              clearTimeout(safetyTimeout);
              done();
            }
          }
        })
        .catch(() => {
          clearTimeout(fetchTimeout);
          // If halted because user paused or changed session, exit!
          if (this.isHalted || this.sessionId !== expectedSession) {
            clearTimeout(safetyTimeout);
            done();
            return;
          }
          // 2. Seamless Instant Fallback: Web Speech API (runs immediately when TTS is unavailable or times out!)
          this.speakWebSpeech(phrase, expectedSession, () => {
            clearTimeout(safetyTimeout);
            done();
          });
        });
    });
  }

  private speakWebSpeech(phrase: string, expectedSession: number, onComplete: () => void): void {
    if (
      this.isHalted ||
      this.sessionId !== expectedSession ||
      typeof window === "undefined" ||
      !("speechSynthesis" in window)
    ) {
      onComplete();
      return;
    }

    try {
      // Circumvent Chromium speech synthesis freeze/pause bug
      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
      }

      const utterance = new SpeechSynthesisUtterance(phrase);
      this.currentUtterance = utterance;
      // Retain active utterance on window so V8 GC does not collect it mid-speech (Chromium bug #338372)
      (window as unknown as { __activeUtterance?: SpeechSynthesisUtterance }).__activeUtterance = utterance;

      utterance.rate = 1.02; // Clear articulate cadence
      utterance.pitch = 1.05; // Elevated penetrating vocal frequency
      utterance.volume = 1.0; // Maximum speech volume

      const voices = window.speechSynthesis.getVoices();
      const preferred = voices.find(
        (v) =>
          v.lang.startsWith("en") &&
          (v.name.includes("David") ||
            v.name.includes("Natural") ||
            v.name.includes("Google") ||
            v.name.includes("Guy") ||
            v.name.includes("Mark") ||
            v.name.includes("Aria"))
      );
      if (preferred) {
        utterance.voice = preferred;
      }

      let finished = false;
      const finish = () => {
        if (!finished) {
          finished = true;
          if (this.currentUtterance === utterance) this.currentUtterance = null;
          (window as unknown as { __activeUtterance?: SpeechSynthesisUtterance | null }).__activeUtterance = null;
          onComplete();
        }
      };

      utterance.onend = finish;
      utterance.onerror = (e) => {
        console.warn("Web Speech utterance error:", e);
        finish();
      };

      // Safety timeout: if browser speech never triggers onend
      setTimeout(finish, 5000);

      window.speechSynthesis.speak(utterance);
      // Double resume for mobile/desktop browsers that queue without speaking
      window.speechSynthesis.resume();
    } catch (e) {
      console.warn("Speech synthesis invocation error:", e);
      onComplete();
    }
  }

  /**
   * Instantly stops any playing voice alert, cancels speech synthesis,
   * pauses HTML5 audio, aborts active requests, and clears pending queue.
   */
  public stop(): void {
    this.sessionId++;
    this.isHalted = true;
    this.isPlaying = false;
    this.queue = [];

    if (this.nextTimer) {
      clearTimeout(this.nextTimer);
      this.nextTimer = null;
    }

    if (this.currentAbortController) {
      try {
        this.currentAbortController.abort();
      } catch {
        // ignore
      }
      this.currentAbortController = null;
    }

    if (this.currentAudio) {
      try {
        this.currentAudio.onended = null;
        this.currentAudio.onerror = null;
        this.currentAudio.pause();
        this.currentAudio.currentTime = 0;
        this.currentAudio.src = "";
      } catch {
        // ignore
      }
      this.currentAudio = null;
    }

    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore
      }
    }

    if (this.currentUtterance) {
      this.currentUtterance.onend = null;
      this.currentUtterance.onerror = null;
      this.currentUtterance = null;
    }
    if (typeof window !== "undefined") {
      (window as unknown as { __activeUtterance?: SpeechSynthesisUtterance | null }).__activeUtterance = null;
    }

    this.onSpeakingChange?.(null);
  }

  public clear(): void {
    this.stop();
  }
}

export const alertEngine = new AlertQueueManager();
