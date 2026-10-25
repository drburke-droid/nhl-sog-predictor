import { useEffect, useState } from "react";
import { ModeA } from "./modes/ModeA";
import { BParams, DEFAULT_B, ModeB } from "./modes/ModeB";
import { CParams, DEFAULT_C, ModeC } from "./modes/ModeC";
import { DEFAULT_D, DParams, ModeD } from "./modes/ModeD";
import { DEFAULT_SHARED, Shared } from "./scenarios";
import { load, save } from "./components/ui";

const TABS = [
  { id: "a", label: "A · Three competing routes" },
  { id: "b", label: "B · Watch drivers find equilibrium" },
  { id: "c", label: "C · Braess's paradox" },
  { id: "d", label: "D · What the app shows" },
] as const;
type TabId = (typeof TABS)[number]["id"];
const KEY = "traffic-lab-v1";

interface Saved {
  tab: TabId;
  shared: Shared;
  b: BParams;
  c: CParams;
  d: DParams;
  theme: "auto" | "light" | "dark";
}

export default function App() {
  const init = load<Saved>(KEY, { tab: "a", shared: DEFAULT_SHARED, b: DEFAULT_B, c: DEFAULT_C, d: DEFAULT_D, theme: "auto" });
  const fromHash = (location.hash.replace("#", "") as TabId) || init.tab;
  const [tab, setTab] = useState<TabId>(TABS.some((t) => t.id === fromHash) ? fromHash : "a");
  const [shared, setShared] = useState<Shared>({ ...DEFAULT_SHARED, ...init.shared });
  const [b, setBState] = useState<BParams>({ ...DEFAULT_B, ...init.b });
  const [c, setCState] = useState<CParams>({ ...DEFAULT_C, ...init.c });
  const [d, setDState] = useState<DParams>({ ...DEFAULT_D, ...init.d });
  const [theme, setTheme] = useState<Saved["theme"]>(init.theme);

  useEffect(() => save(KEY, { tab, shared, b, c, d, theme }), [tab, shared, b, c, d, theme]);
  useEffect(() => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  useEffect(() => {
    try {
      history.replaceState(null, "", `#${tab}`);
    } catch {
      /* some embedded contexts refuse history changes; the tab still works */
    }
  }, [tab]);
  useEffect(() => {
    const onHash = () => {
      const h = location.hash.replace("#", "") as TabId;
      if (TABS.some((t) => t.id === h)) setTab(h);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const set = (p: Partial<Shared>) => setShared((s) => ({ ...s, ...p }));
  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Traffic Equilibrium Lab</h1>
          <p>Why do completely different routes so often show the same ETA? Drivers crowd the quicker roads until no one gains by switching, and apps show only the close calls. Explore both. <span className="badge-toy">toy models, not traffic forecasts</span></p>
        </div>
        <button className="theme-btn" onClick={() => setTheme(theme === "auto" ? "dark" : theme === "dark" ? "light" : "auto")} aria-label="Colour theme">
          Theme: {theme}
        </button>
      </header>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      {tab === "a" && <ModeA s={shared} set={set} />}
      {tab === "b" && <ModeB s={shared} set={set} b={b} setB={(p) => setBState((x) => ({ ...x, ...p }))} />}
      {tab === "c" && <ModeC c={c} setC={(p) => setCState((x) => ({ ...x, ...p }))} />}
      {tab === "d" && <ModeD d={d} setD={(p) => setDState((x) => ({ ...x, ...p }))} />}
      <footer className="foot">
        Models: Wardrop user equilibrium and system optimum on independent parallel routes with the BPR curve (exact, by bisection);
        a seeded day-to-day learning process; user equilibrium on the Braess network by path equilibration. Units: minutes, and vehicles per hour over one steady peak hour.
        The New York and Calgary free-flow times come from OpenStreetMap road data via Overture Maps (© OpenStreetMap contributors, ODbL) at the speed limit, with no traffic; everything else is illustrative.
        {" "}Settings are saved in this browser only. <button className="btn" onClick={() => { try { localStorage.removeItem(KEY); } catch { /* ignore */ } location.reload(); }}>Reset everything</button>
      </footer>
    </div>
  );
}
