import { resolveAsset } from "../api";

export default function ResultPage({ result }) {
  if (!result) {
    return (
      <section className="panel">
        <h2>Results</h2>
        <p className="muted">Run an analysis to generate forensic outputs.</p>
      </section>
    );
  }

  const verdict = result.verdict || "UNKNOWN";
  const isGenuine = verdict === "GENUINE";
  const metrics = result.metrics || {};

  return (
    <section className="panel">
      <h2>Forensic Decision</h2>

      <div className={`verdict-banner ${isGenuine ? "ok" : "bad"}`}>
        {verdict}
      </div>

      <div className="metrics-grid">
        <article>
          <h4>Z-score</h4>
          <p>{metrics.z_score ?? "-"}</p>
        </article>
        <article>
          <h4>Combined Score</h4>
          <p>{metrics.combined_score ?? "-"}</p>
        </article>
        <article>
          <h4>Vault Mean</h4>
          <p>{metrics.vault_mean ?? "-"}</p>
        </article>
        <article>
          <h4>Vault Std</h4>
          <p>{metrics.vault_std ?? "-"}</p>
        </article>
      </div>

      <div className="xai-block">
        <h3>Backbone Attention (Grad-CAM)</h3>
        <p className="muted">Evidence map used for forensic review and explainability audit.</p>
        {result.artifacts?.grad_cam_report ? (
          <img src={resolveAsset(result.artifacts.grad_cam_report)} alt="Grad-CAM report" className="xai-image" />
        ) : (
          <p className="muted">No Grad-CAM artifact available.</p>
        )}
      </div>
    </section>
  );
}
