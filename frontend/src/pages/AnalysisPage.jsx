import { resolveAsset } from "../api";

const LABELS = ["Shape", "Pseudo-Pressure", "Stroke Angle", "Skeleton"];

export default function AnalysisPage({ progress, channelImages }) {
  const slots = [0, 1, 2, 3];

  return (
    <section className="panel">
      <h2>Tensor Analysis</h2>
      <p className="muted">4-channel forensic tensor derived from the questioned signature.</p>

      <div className="analysis-status">
        <h3>Current Stage</h3>
        <p>{progress.stage || "queued"}</p>
        <p className="muted">{progress.detail || "Awaiting updates..."}</p>
      </div>

      <div className="channel-grid">
        {slots.map((idx) => {
          const src = channelImages[idx] ? resolveAsset(channelImages[idx]) : "";
          return (
            <article className="channel-card" key={LABELS[idx]}>
              <h4>{LABELS[idx]}</h4>
              {src ? <img src={src} alt={LABELS[idx]} /> : <div className="channel-placeholder">Generating channel...</div>}
            </article>
          );
        })}
      </div>
    </section>
  );
}
