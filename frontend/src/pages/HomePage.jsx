export default function HomePage({
  vaultFiles,
  questionedFile,
  onVaultChange,
  onQuestionedChange,
  onSubmit,
  isRunning,
  error,
}) {
  return (
    <section className="panel">
      <h2>Case Intake</h2>
      <p className="muted">
        Upload a minimum of 5 genuine reference signatures into the Vault and one questioned signature for forensic verification.
      </p>

      <form className="upload-form" onSubmit={onSubmit}>
        <label className="field-label">
          Vault References (5+ files)
          <input
            type="file"
            accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp,.npy"
            multiple
            onChange={onVaultChange}
            disabled={isRunning}
          />
        </label>

        <div className="file-list">
          {vaultFiles.length > 0 ? (
            vaultFiles.map((file) => <span key={`${file.name}-${file.lastModified}`}>{file.name}</span>)
          ) : (
            <span className="muted">No vault files selected yet.</span>
          )}
        </div>

        <label className="field-label">
          Questioned Signature (single file)
          <input
            type="file"
            accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp,.npy"
            onChange={onQuestionedChange}
            disabled={isRunning}
          />
        </label>

        <div className="file-list">
          {questionedFile ? <span>{questionedFile.name}</span> : <span className="muted">No questioned file selected.</span>}
        </div>

        {error ? <p className="error-text">{error}</p> : null}

        <button className="primary-btn" type="submit" disabled={isRunning}>
          {isRunning ? "Processing..." : "Start Forensic Analysis"}
        </button>
      </form>
    </section>
  );
}
