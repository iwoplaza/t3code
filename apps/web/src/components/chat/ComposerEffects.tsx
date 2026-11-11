import { perlin3d } from "@typegpu/noise";
import { sdRoundedBox2d } from "@typegpu/sdf";
import { useEffect, useImperativeHandle, useRef, type Ref } from "react";
import { common, d, std, tgpu, type TgpuRoot } from "typegpu";

import { observeResize } from "~/lib/observeResize";

// Room around the composer the glow can spread into, in CSS px. Matches `-inset-14`.
const PAD = 56;
const RADIUS = 16;
const TAU = Math.PI * 2;
const NEVER = 1e3;

// Every keystroke gets a ripple and a glow at the edge nearest the caret; overlaps add up.
const PULSE_SLOTS = 12;
const PULSE_SECONDS = 0.35;
const PULSE_SPEED = 60;
const PULSE_STRENGTH = 0.3;
// How far along the edge, in px, a ripple stays visible either side of the key.
const PULSE_REACH = 80;
const GLOW_SECONDS = 0.55;
const GLOW_RADIUS = 22;
const GLOW_STRENGTH = 0.09;
// How far, in px, the noise bends a ripple by the end of its life, and how many bends it has.
const PULSE_WARP = 32;
const PULSE_WARP_FREQUENCY = 4;
const PULSE_DRIFT = 3;
const SEND_SECONDS = 0.5;
// How far, in radians, the faintest parts of a keystroke pulse rotate away from the theme hue.
const PULSE_HUE_SHIFT = 0.7;
const COMPLETE_SECONDS = 1.5;
const ATTENTION_PULSE_SECONDS = 0.75;
const ATTENTION_PULSES = 3;
// Perlin gradients are precomputed for this domain; samples outside it wrap around.
const NOISE_CACHE_SIZE = 32;
// Two octaves of Perlin noise never add up to more than this.
const MAX_WARP = 1.5;
// The glow is soft, so drawing it at device resolution buys nothing but 4x the pixels on 2x screens.
const RENDER_SCALE = 1;

const Params = d.struct({
  // The theme's primary colour; every effect uses it and differs only in shape and motion.
  color: d.vec3f,
  size: d.vec2f,
  boxHalf: d.vec2f,
  radius: d.f32,
  // Live keystroke pulses come first: their age, noise seed, and origin relative to the centre.
  pulseCount: d.u32,
  pulses: d.arrayOf(d.f32, PULSE_SLOTS),
  seeds: d.arrayOf(d.f32, PULSE_SLOTS),
  origins: d.arrayOf(d.vec2f, PULSE_SLOTS),
  sendAge: d.f32,
  completeAge: d.f32,
  attentionAge: d.f32,
});

// Storage, not uniform: uniform arrays need a 16-byte stride, which f32 and vec2f elements lack.
const layout = tgpu.bindGroupLayout({ params: { storage: Params } });

/** Rotates a colour around the grey axis, which turns its hue and keeps its brightness. */
function hueRotate(color: d.v3f, angle: number) {
  "use gpu";
  const axis = d.vec3f(0.57735);
  const c = std.cos(angle);
  const parallel = axis.mul(std.dot(axis, color) * (1 - c));
  const turned = color.mul(c).add(std.cross(axis, color).mul(std.sin(angle)));
  return std.saturate(turned.add(parallel));
}

function gaussian(x: number, width: number) {
  "use gpu";
  const t = x / width;
  return std.exp(-t * t);
}

function strokePulse(
  p: d.v2f,
  outside: number,
  warpAt: d.v2f,
  origin: d.v2f,
  age: number,
  seed: number,
) {
  "use gpu";
  if (age >= GLOW_SECONDS) {
    return d.f32(0);
  }
  const q = p.sub(origin);
  const glowT = age / GLOW_SECONDS;
  const glow = GLOW_STRENGTH * gaussian(std.length(q), GLOW_RADIUS) * (1 - glowT) * (1 - glowT);
  if (age >= PULSE_SECONDS) {
    return glow;
  }
  const t = age / PULSE_SECONDS;
  const ringWidth = 1.5 + age * 4;
  const reachWidth = PULSE_REACH * (0.4 + t);
  // Skip the noise for pixels the ripple can't reach, however the noise bends it.
  const maxBend = MAX_WARP * (1 + age * PULSE_WARP);
  const alongEdge = std.sqrt(std.max(std.dot(q, q) - outside * outside, 0));
  if (
    std.abs(outside - age * PULSE_SPEED) > maxBend + 3 * ringWidth ||
    alongEdge > 3 * reachWidth
  ) {
    return glow;
  }
  const fade = (1 - t) * (1 - t) * (1 - t);
  // Two octaves: broad bends plus finer, faster churn on top.
  const z = seed + age * PULSE_DRIFT;
  const warp =
    perlin3d.sample(d.vec3f(warpAt, z)) +
    0.5 * perlin3d.sample(d.vec3f(warpAt.mul(2.3), z * 1.7 + 19));
  // The ripple is a copy of the outline pushed outwards, visible only along the edge near the key.
  const offset = age * PULSE_SPEED + warp * (1 + age * PULSE_WARP);
  const ring = gaussian(outside - offset, ringWidth);
  const reach = gaussian(alongEdge, reachWidth);
  return glow + PULSE_STRENGTH * ring * reach * fade;
}

const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
  "use gpu";
  const params = layout.$.params;
  const p = uv.sub(0.5).mul(params.size);
  const dist = sdRoundedBox2d(p, params.boxHalf, params.radius);
  if (dist < -3) {
    return d.vec4f();
  }
  const outside = std.max(dist, 0);
  // Fades everything out just inside the border so the glow never sits on the text.
  const rim = std.smoothstep(-3, 1, dist);

  // Position around the perimeter: 0 at top centre, ±0.5 at bottom centre.
  const around = std.atan2(p.x / params.boxHalf.x, -p.y / params.boxHalf.y) / TAU;
  const perimeter = 4 * (params.boxHalf.x + params.boxHalf.y) - (8 - TAU) * params.radius;

  // 1. Typing: each keystroke leaves a ripple and a lingering glow at its spot on the edge.
  // Noise is read on a circle around the composer, so the warp wraps without a seam.
  const warpAt = d.vec2f(std.cos(around * TAU), std.sin(around * TAU)).mul(PULSE_WARP_FREQUENCY);
  let pulses = d.f32(0);
  for (let i = d.u32(0); i < params.pulseCount; i++) {
    pulses += strokePulse(
      p,
      outside,
      warpAt,
      params.origins[i]!,
      params.pulses[i]!,
      params.seeds[i]!,
    );
  }
  const strokeA = pulses * rim;

  // 2. Send shockwave: one ring leaves the composer and dissolves.
  const sendT = std.saturate(params.sendAge / SEND_SECONDS);
  const sendFade = (1 - sendT) * (1 - sendT);
  const sendRing = gaussian(outside - params.sendAge * 80, 1.5 + params.sendAge * 6) * sendFade;
  const sendFlash = std.exp(-outside / 4) * std.exp(-params.sendAge * 12);
  const sendA = (0.8 * sendRing + 0.6 * sendFlash) * rim;

  // 3. Turn finished: two comets run from the top centre down both sides and meet at the bottom.
  const head = 0.5 * std.smoothstep(0, 0.9, params.completeAge);
  const behind = (head - std.abs(around)) * perimeter;
  const trail = std.select(0, std.exp(-behind / 90), behind > 0);
  const comet = gaussian(behind, 14) + 0.45 * trail;
  const completeFade = 1 - std.smoothstep(0.8, COMPLETE_SECONDS, params.completeAge);
  const completeA = comet * std.exp(-outside / 5) * completeFade * rim;

  // 4. Needs you: a light on the top edge, toward the request above, breathes a few times.
  const pulsePhase = params.attentionAge / ATTENTION_PULSE_SECONDS;
  const breath = std.sin(std.fract(pulsePhase) * Math.PI);
  const pulse = std.select(0, breath * breath, pulsePhase < ATTENTION_PULSES);
  const topEdge = gaussian(around * perimeter, perimeter * 0.12);
  const attentionA = 0.8 * pulse * topEdge * std.exp(-outside / 10) * rim;

  const strokeHue = PULSE_HUE_SHIFT * (1 - std.min(strokeA / PULSE_STRENGTH, 1));
  const strokeColor = hueRotate(params.color, strokeHue);
  const otherA = sendA + completeA + attentionA;
  const total = strokeA + otherA;
  const alpha = std.min(total, 1);
  const mixed = strokeColor.mul(strokeA).add(params.color.mul(otherA));
  const rgb = mixed.mul(alpha / std.max(total, 1e-4));
  return d.vec4f(rgb, alpha);
});

let rootPromise: Promise<TgpuRoot | null> | null = null;
const pipelines = new WeakMap<TgpuRoot, ReturnType<typeof createPipeline>>();

function createPipeline(root: TgpuRoot) {
  const size = NOISE_CACHE_SIZE;
  const noiseCache = perlin3d.staticCache({ root, size: d.vec3u(size, size, size) });
  return root
    .pipe(noiseCache.inject())
    .createRenderPipeline({ vertex: common.fullScreenTriangle, fragment });
}

function getRoot(): Promise<TgpuRoot | null> {
  if (rootPromise) return rootPromise;
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    rootPromise = Promise.resolve(null);
    return rootPromise;
  }
  rootPromise = tgpu.init().then(
    (root) => {
      void root.device.lost.then(() => {
        rootPromise = null;
      });
      return root;
    },
    () => null,
  );
  return rootPromise;
}

let colorProbe: CanvasRenderingContext2D | null = null;

/** Resolves any CSS colour (oklch, color-mix, ...) to sRGB in [0, 1]. */
function cssColorToRgb(color: string): [number, number, number] {
  colorProbe ??= document.createElement("canvas").getContext("2d", { willReadFrequently: true });
  if (!colorProbe) return [0.5, 0.5, 1];
  colorProbe.clearRect(0, 0, 1, 1);
  colorProbe.fillStyle = color;
  colorProbe.fillRect(0, 0, 1, 1);
  const [r = 0, g = 0, b = 0] = colorProbe.getImageData(0, 0, 1, 1).data;
  return [r / 255, g / 255, b / 255];
}

function caretRect(host: Element | null): DOMRect | null {
  const selection = window.getSelection();
  if (!host || !selection || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  if (!host.contains(range.startContainer)) return null;
  // A collapsed range at the start of an empty line has no client rects; fall back to its node.
  const rect = range.getClientRects()[0];
  if (rect) return rect;
  const node = range.startContainer;
  return (node instanceof Element ? node : node.parentElement)?.getBoundingClientRect() ?? null;
}

const ORIGIN: [number, number] = [0, 0];

function pad<T>(values: T[], filler: T): T[] {
  return Array.from({ length: PULSE_SLOTS }, (_, index) => values[index] ?? filler);
}

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

interface Engine {
  stroke(): void;
  send(): void;
  complete(): void;
  attention(): void;
  dispose(): void;
}

function createEngine(canvas: HTMLCanvasElement): Engine {
  let disposed = false;
  let frame: number | null = null;
  let boxWidth = 0;
  let boxHeight = 0;
  let color: [number, number, number] = [0.5, 0.5, 1];
  let strokeAt = -NEVER;
  const pulseTimes = Array.from({ length: PULSE_SLOTS }, () => -NEVER);
  const pulseSeeds = Array.from({ length: PULSE_SLOTS }, () => 0);
  const pulseOrigins = Array.from({ length: PULSE_SLOTS }, () => [0, 0] as [number, number]);
  let pulseCursor = 0;
  let sendAt = -NEVER;
  let completeAt = -NEVER;
  let attentionAt = -NEVER;
  let gpu: {
    root: TgpuRoot;
    context: GPUCanvasContext;
    params: ReturnType<typeof createParams>;
    bindGroup: ReturnType<TgpuRoot["createBindGroup"]>;
    pipeline: ReturnType<typeof createPipeline>;
  } | null = null;

  function createParams(root: TgpuRoot) {
    return root.createBuffer(Params).$usage("storage");
  }

  const resize = () => {
    const host = canvas.parentElement;
    if (!host) return;
    boxWidth = host.offsetWidth;
    boxHeight = host.offsetHeight;
    canvas.width = Math.max(1, Math.round((boxWidth + PAD * 2) * RENDER_SCALE));
    canvas.height = Math.max(1, Math.round((boxHeight + PAD * 2) * RENDER_SCALE));
  };
  const stopObserving = canvas.parentElement ? observeResize(canvas.parentElement, resize) : null;
  resize();

  void getRoot().then((root) => {
    if (!root || disposed) return;
    const context = root.configureContext({ canvas, alphaMode: "premultiplied" });
    const params = createParams(root);
    const bindGroup = root.createBindGroup(layout, { params });
    let pipeline = pipelines.get(root);
    if (!pipeline) {
      pipeline = createPipeline(root);
      pipelines.set(root, pipeline);
    }
    gpu = { root, context, params, bindGroup, pipeline };
  });

  const render = (now: number) => {
    frame = null;
    if (!gpu || disposed) return;
    const seconds = now / 1000;

    const strokeAge = seconds - strokeAt;
    // Only pulses still glowing go to the GPU, so idle slots cost nothing per pixel.
    const live = pulseTimes.flatMap((at, slot) => (seconds - at < GLOW_SECONDS ? [slot] : []));
    const sendAge = seconds - sendAt;
    const completeAge = seconds - completeAt;
    const attentionAge = seconds - attentionAt;
    gpu.params.write({
      color,
      size: d.vec2f(boxWidth + PAD * 2, boxHeight + PAD * 2),
      boxHalf: d.vec2f(boxWidth / 2, boxHeight / 2),
      radius: RADIUS,
      pulseCount: live.length,
      pulses: pad(
        live.map((slot) => seconds - (pulseTimes[slot] ?? 0)),
        NEVER,
      ),
      seeds: pad(
        live.map((slot) => pulseSeeds[slot] ?? 0),
        0,
      ),
      origins: pad(
        live.map((slot) => pulseOrigins[slot] ?? ORIGIN),
        ORIGIN,
      ),
      sendAge: Math.min(sendAge, NEVER),
      completeAge: Math.min(completeAge, NEVER),
      attentionAge: Math.min(attentionAge, NEVER),
    });
    gpu.pipeline
      .with(gpu.bindGroup)
      .withColorAttachment({
        view: gpu.context,
        loadOp: "clear",
        storeOp: "store",
        clearValue: [0, 0, 0, 0],
      })
      .draw(3);

    // Render only while something is visible; an idle composer costs nothing.
    const active =
      strokeAge < GLOW_SECONDS ||
      sendAge < SEND_SECONDS ||
      completeAge < COMPLETE_SECONDS ||
      attentionAge < ATTENTION_PULSE_SECONDS * ATTENTION_PULSES;
    if (active) frame = requestAnimationFrame(render);
  };

  // The point on the composer edge closest to the caret, relative to the composer's centre.
  const nearestEdgeToCaret = (): [number, number] => {
    const hx = boxWidth / 2;
    const hy = boxHeight / 2;
    const caret = caretRect(canvas.parentElement);
    const host = canvas.parentElement?.getBoundingClientRect();
    if (!caret || !host) return [0, -hy];
    const x = Math.min(Math.max(caret.left - host.left - hx, -hx), hx);
    const y = Math.min(Math.max(caret.top + caret.height / 2 - host.top - hy, -hy), hy);
    const toLeft = x + hx;
    const toRight = hx - x;
    const toTop = y + hy;
    const toBottom = hy - y;
    const nearest = Math.min(toLeft, toRight, toTop, toBottom);
    if (nearest === toTop) return [x, -hy];
    if (nearest === toBottom) return [x, hy];
    if (nearest === toLeft) return [-hx, y];
    return [hx, y];
  };

  const trigger = (apply: (seconds: number) => void) => {
    if (disposed || !gpu || prefersReducedMotion()) return;
    apply(performance.now() / 1000);
    if (frame !== null) return;
    // Read per animation, not per frame, so theme switches apply without an observer.
    color = cssColorToRgb(getComputedStyle(canvas).color);
    frame = requestAnimationFrame(render);
  };

  return {
    stroke: () =>
      trigger((seconds) => {
        strokeAt = seconds;
        pulseTimes[pulseCursor] = seconds;
        pulseSeeds[pulseCursor] = Math.random() * 256;
        pulseOrigins[pulseCursor] = nearestEdgeToCaret();
        pulseCursor = (pulseCursor + 1) % PULSE_SLOTS;
      }),
    send: () =>
      trigger((seconds) => {
        pulseTimes.fill(-NEVER);
        sendAt = seconds;
      }),
    complete: () =>
      trigger((seconds) => {
        completeAt = seconds;
      }),
    attention: () =>
      trigger((seconds) => {
        attentionAt = seconds;
      }),
    dispose: () => {
      disposed = true;
      stopObserving?.();
      if (frame !== null) cancelAnimationFrame(frame);
      if (gpu) {
        gpu.context.unconfigure();
        gpu.params.destroy();
        gpu = null;
      }
    },
  };
}

export interface ComposerEffectsHandle {
  stroke(): void;
  send(): void;
}

/**
 * WebGPU glow drawn behind the composer card. Place it inside a `relative` box sized to the card.
 * Renders nothing (and costs nothing) without WebGPU, with reduced motion, or between effects.
 */
export function ComposerEffects({
  ref,
  scopeKey,
  running,
  needsAttention,
}: {
  ref: Ref<ComposerEffectsHandle>;
  /** Identity of what the composer is showing; transitions across a change don't animate. */
  scopeKey: string;
  running: boolean;
  needsAttention: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const previousRef = useRef({ scopeKey, running, needsAttention });

  useImperativeHandle(
    ref,
    () => ({
      stroke: () => engineRef.current?.stroke(),
      send: () => engineRef.current?.send(),
    }),
    [],
  );

  // oxlint-disable-next-line t3code/no-unreasoned-state-sync -- owns the WebGPU canvas context and its render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const engine = createEngine(canvas);
    engineRef.current = engine;
    return () => {
      engine.dispose();
      engineRef.current = null;
    };
  }, []);

  // oxlint-disable-next-line t3code/no-unreasoned-state-sync -- turn-state edges start one-shot GPU animations
  useEffect(() => {
    const previous = previousRef.current;
    previousRef.current = { scopeKey, running, needsAttention };
    if (previous.scopeKey !== scopeKey) return;
    if (needsAttention && !previous.needsAttention) {
      engineRef.current?.attention();
    } else if (previous.running && !running && !needsAttention) {
      engineRef.current?.complete();
    }
  }, [scopeKey, running, needsAttention]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none absolute -inset-14 z-0 h-[calc(100%+7rem)] w-[calc(100%+7rem)] text-primary"
    />
  );
}
