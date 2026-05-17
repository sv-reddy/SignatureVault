import { useMemo, useState } from "react";
import { fetchProgress, startAnalysis } from "./api";
import HomePage from "./pages/HomePage";
import AnalysisPage from "./pages/AnalysisPage";
import ResultPage from "./pages/ResultPage";
import ProgressBar from "./components/ProgressBar";

function newRequestId() {
  if (window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID();
  }
  return `req-${Date.now()}`;
}

export default function App() {
  const [vaultFiles, setVaultFiles] = useState([]);
  const [questionedFile, setQuestionedFile] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");
  const [activePage, setActivePage] = useState("home");
  const [progress, setProgress] = useState({ stage: "idle", percent: 0, detail: "" });
  const [result, setResult] = useState(null);

  const channelImages = useMemo(() => result?.artifacts?.channel_images || [], [result]);

  const onVaultChange = (event) => {
    setVaultFiles(Array.from(event.target.files || []));
    setError("");
  };

  const onQuestionedChange = (event) => {
    const file = event.target.files?.[0] || null;
    setQuestionedFile(file);
    setError("");
  };

  const onSubmit = async (event) => {
    event.preventDefault();

    if (vaultFiles.length < 5) {
      setError("At least 5 vault images are required.");
      return;
    }
    if (!questionedFile) {
      setError("A questioned signature file is required.");
      return;
    }

    const requestId = newRequestId();
    setIsRunning(true);
    setError("");
    setResult(null);
    setActivePage("analysis");
    setProgress({ stage: "queued", percent: 5, detail: "Preparing request" });

    const poller = window.setInterval(async () => {
      try {
        const p = await fetchProgress(requestId);
        setProgress(p);
      } catch (_) {
        // Progress may be briefly unavailable before backend initializes the job.
      }
    }, 900);

    try {
      const data = await startAnalysis({
        requestId,
        vaultFiles,
        questionedFile,
      });
      setResult(data);
      setProgress({ stage: "completed", percent: 100, detail: "Pipeline complete" });
      setActivePage("result");
    } catch (err) {
      setError(err.message || "Analysis failed");
      setActivePage("home");
    } finally {
      window.clearInterval(poller);
      setIsRunning(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="top-banner">
        <div>
          <h1>SignatureVault</h1>
          <p>Offline Signature Forensics Interface</p>
        </div>
        <nav className="top-nav">
          <button className={activePage === "home" ? "active" : ""} onClick={() => setActivePage("home")}>
            Home
          </button>
          <button className={activePage === "analysis" ? "active" : ""} onClick={() => setActivePage("analysis")}>
            Analysis
          </button>
          <button className={activePage === "result" ? "active" : ""} onClick={() => setActivePage("result")}>
            Result
          </button>
        </nav>
      </header>

      <ProgressBar percent={progress.percent} label={progress.detail} />

      {activePage === "home" ? (
        <HomePage
          vaultFiles={vaultFiles}
          questionedFile={questionedFile}
          onVaultChange={onVaultChange}
          onQuestionedChange={onQuestionedChange}
          onSubmit={onSubmit}
          isRunning={isRunning}
          error={error}
        />
      ) : null}

      {activePage === "analysis" ? <AnalysisPage progress={progress} channelImages={channelImages} /> : null}
      {activePage === "result" ? <ResultPage result={result} /> : null}
    </main>
  );
}
