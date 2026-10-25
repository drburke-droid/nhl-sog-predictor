import { ReactNode, useId, useState } from "react";

export const ROUTE_VAR = ["var(--r0)", "var(--r1)", "var(--r2)"];
export const ROUTE_DASH = ["", "9 5", "2 4"]; // solid / dashed / dotted: never colour alone
export const SWATCH_CLASS = ["", "dash", "dot"];
export const LETTERS = ["A", "B", "C"];

export const fmt = (x: number, d = 1) => (Number.isFinite(x) ? x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—");
export const fmtInt = (x: number) => (Number.isFinite(x) ? Math.round(x).toLocaleString("en-US") : "—");

export function Slider(props: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: ReactNode;
}) {
  const id = useId();
  const f = props.format ?? ((v: number) => String(v));
  return (
    <div className="slider">
      <label htmlFor={id}>{props.label}</label>
      <output htmlFor={id}>{f(props.value)}</output>
      <input id={id} type="range" min={props.min} max={props.max} step={props.step} value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))} />
      {props.hint && <div className="hint">{props.hint}</div>}
    </div>
  );
}

export const TERMS: Record<string, string> = {
  equilibrium:
    "User equilibrium (Wardrop's first principle, 1952): no driver can get there sooner by switching routes on their own, given what everyone else does. Every route that carries traffic then has the same travel time; a route nobody uses is no faster.",
  "system optimum":
    "System optimum (Wardrop's second principle): the split of traffic that minimizes the total time everyone spends travelling. It equalizes each route's marginal social cost (your own time plus the delay you add to others), not the travel times themselves, so some drivers get slower trips than others.",
  "price of anarchy":
    "Total travel time at user equilibrium divided by total travel time at the system optimum. 1.00 means selfish route choice costs nothing; 1.10 would mean 10 % more vehicle-minutes than the best possible split.",
  "marginal social cost":
    "The extra total travel time caused by one more car on a route: its own trip time plus the small delay it adds to every other car already there.",
  BPR:
    "The Bureau of Public Roads curve (1964): time = free-flow time × (1 + α (flow/capacity)^β). A smooth, stylized congestion curve used in planning models. It cannot represent queues backing up, spillback or gridlock.",
};

export function Term({ k, children }: { k: keyof typeof TERMS | string; children?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <button type="button" className="term" aria-describedby={id} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}>
      {children ?? k}
      <span id={id} role="tooltip" className="pop" hidden={!open}>{TERMS[k]}</span>
    </button>
  );
}

export function RouteKey({ i, name }: { i: number; name: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span className="letter" style={{ background: ROUTE_VAR[i] }}>{LETTERS[i]}</span>
      <span className={`swatch ${SWATCH_CLASS[i]}`} style={{ color: ROUTE_VAR[i] }} aria-hidden />
      <span>{name}</span>
    </span>
  );
}

export function Tile({ k, v, s }: { k: ReactNode; v: ReactNode; s?: ReactNode }) {
  return (
    <div className="tile">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
      {s && <div className="s">{s}</div>}
    </div>
  );
}

export function load<T>(key: string, fallback: T): T {
  try {
    const s = localStorage.getItem(key);
    if (!s) return fallback;
    return { ...fallback, ...JSON.parse(s) };
  } catch {
    return fallback;
  }
}

export function save(key: string, v: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(v));
  } catch {
    /* storage can be unavailable (private windows); settings just won't persist */
  }
}
