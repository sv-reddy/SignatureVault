export default function ProgressBar({ percent = 0, label = "" }) {
  const safePercent = Math.max(0, Math.min(100, Number(percent || 0)));

  return (
    <div className="progress-wrap" aria-label="Analysis progress">
      <div className="progress-header">
        <span>Pipeline Progress</span>
        <span>{safePercent}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${safePercent}%` }} />
      </div>
      <p className="progress-label">{label || "Waiting for backend stage updates..."}</p>
    </div>
  );
}
