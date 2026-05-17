const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { randomUUID } = require("crypto");

const app = express();
const PORT = process.env.PORT || 5000;

const BACKEND_ROOT = __dirname;
const PROJECT_ROOT = path.resolve(BACKEND_ROOT, "..");
const RESULTS_ROOT = path.join(BACKEND_ROOT, "results");
const TMP_ROOT = path.join(BACKEND_ROOT, "tmp", "api_jobs");
const CHECKPOINT_PATH = path.join(BACKEND_ROOT, "checkpoints", "best_tavnet.pt");

const ALLOWED_EXTS = new Set([".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".npy"]);
const jobProgress = new Map();

fs.mkdirSync(TMP_ROOT, { recursive: true });

app.use(cors());
app.use(express.json());
app.use("/static", express.static(RESULTS_ROOT));

function setProgress(jobId, stage, percent, detail) {
  jobProgress.set(jobId, {
    jobId,
    stage,
    percent,
    detail,
    updatedAt: new Date().toISOString(),
  });
}

function ensureImageExtension(fileName) {
  const ext = path.extname(fileName || "").toLowerCase();
  return ALLOWED_EXTS.has(ext);
}

function toStaticUrl(absPath) {
  const relFromResults = path.relative(RESULTS_ROOT, absPath);
  if (!relFromResults.startsWith("..")) {
    return `/static/${relFromResults.replace(/\\/g, "/")}`;
  }
  return null;
}

function parseJsonFromOutput(text) {
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    try {
      return JSON.parse(lines[i]);
    } catch (_) {
      // Continue searching upward for a JSON line.
    }
  }
  throw new Error("Could not parse JSON output from Python process");
}

function runPython(scriptName, args, onStdoutLine) {
  return new Promise((resolve, reject) => {
    const cmd = process.env.PYTHON_BIN || "python";
    const child = spawn(cmd, [path.join(BACKEND_ROOT, scriptName), ...args], {
      cwd: BACKEND_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      if (onStdoutLine) {
        text.split(/\r?\n/).forEach((line) => {
          if (line.trim()) {
            onStdoutLine(line.trim());
          }
        });
      }
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (err) => {
      reject(new Error(`Failed to start ${scriptName}: ${err.message}`));
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${scriptName} exited with code ${code}. ${stderr || stdout}`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

const upload = multer({
  dest: path.join(BACKEND_ROOT, "tmp", "uploads"),
  limits: {
    fileSize: 15 * 1024 * 1024,
    files: 20,
  },
  fileFilter: (req, file, cb) => {
    if (!ensureImageExtension(file.originalname)) {
      cb(new Error("Invalid image format. Supported: png, jpg, jpeg, bmp, tif, tiff, webp, npy"));
      return;
    }
    cb(null, true);
  },
});

app.get("/api/health", (req, res) => {
  res.json({ ok: true, api: "SignatureVault API Wrapper" });
});

app.get("/api/progress/:jobId", (req, res) => {
  const data = jobProgress.get(req.params.jobId);
  if (!data) {
    res.status(404).json({ error: "Job not found" });
    return;
  }
  res.json(data);
});

app.post(
  "/api/analyze",
  upload.fields([
    { name: "vault", maxCount: 20 },
    { name: "questioned", maxCount: 1 },
  ]),
  async (req, res) => {
    const jobId = String(req.body.requestId || randomUUID());
    const files = req.files || {};
    const vaultFiles = files.vault || [];
    const questionedFiles = files.questioned || [];

    const jobDir = path.join(TMP_ROOT, jobId);
    const vaultDir = path.join(jobDir, "vault");
    const questionedDir = path.join(jobDir, "questioned");

    try {
      setProgress(jobId, "queued", 5, "Request received");

      if (!fs.existsSync(CHECKPOINT_PATH)) {
        throw new Error(`Missing checkpoint: ${CHECKPOINT_PATH}`);
      }

      if (vaultFiles.length < 5) {
        throw new Error("Vault must contain at least 5 reference signature images.");
      }
      if (questionedFiles.length !== 1) {
        throw new Error("Exactly one questioned signature file is required.");
      }

      setProgress(jobId, "validation", 15, "Input validation complete");

      fs.mkdirSync(vaultDir, { recursive: true });
      fs.mkdirSync(questionedDir, { recursive: true });

      for (const file of vaultFiles) {
        if (!ensureImageExtension(file.originalname)) {
          throw new Error(`Invalid vault file format: ${file.originalname}`);
        }
        const target = path.join(vaultDir, path.basename(file.originalname));
        fs.copyFileSync(file.path, target);
      }

      for (const file of questionedFiles) {
        if (!ensureImageExtension(file.originalname)) {
          throw new Error(`Invalid questioned file format: ${file.originalname}`);
        }
        const target = path.join(questionedDir, path.basename(file.originalname));
        fs.copyFileSync(file.path, target);
      }

      const questionedPath = path.join(questionedDir, path.basename(questionedFiles[0].originalname));
      setProgress(jobId, "verify_vault", 35, "Running verification model");

      const verifyArgs = [
        "--vault", vaultDir,
        "--questioned", questionedPath,
        "--checkpoint", CHECKPOINT_PATH,
        "--json-stdout",
      ];
      const verifyRun = await runPython("verify_vault.py", verifyArgs, (line) => {
        if (line.toLowerCase().includes("processing") || line.toLowerCase().includes("verifying")) {
          setProgress(jobId, "verify_vault", 45, line);
        }
      });

      const verifyPayload = parseJsonFromOutput(verifyRun.stdout);
      if (verifyPayload.error) {
        throw new Error(verifyPayload.error);
      }

      setProgress(jobId, "generate_heatmap", 70, "Generating Grad-CAM and feature visualizations");

      const heatmapArgs = [
        "--sample-dir", jobDir,
        "--questioned", questionedPath,
        "--checkpoint", CHECKPOINT_PATH,
        "--out-dir", path.join(RESULTS_ROOT, "grad_cam"),
        "--feature-out-dir", path.join(RESULTS_ROOT, "visualize_features"),
      ];
      const heatmapRun = await runPython("generate_heatmap.py", heatmapArgs, (line) => {
        if (line.toLowerCase().includes("saved") || line.toLowerCase().includes("processing")) {
          setProgress(jobId, "generate_heatmap", 85, line);
        }
      });

      const heatmapPayload = parseJsonFromOutput(heatmapRun.stdout);
      if (heatmapPayload.error) {
        throw new Error(heatmapPayload.error);
      }

      const result = (verifyPayload.results && verifyPayload.results[0]) || null;
      if (!result) {
        throw new Error("Verification produced no result payload");
      }

      const featureGridAbs = heatmapPayload.feature_visualization?.grid;
      const channelAbs = heatmapPayload.feature_visualization?.channels || [];
      const gradCamAbs = heatmapPayload.grad_cam_report;

      const responsePayload = {
        jobId,
        verdict: result.verdict,
        metrics: {
          z_score: result.z_score,
          combined_score: result.combined_score,
          vault_mean: result.vault_mean,
          vault_std: result.vault_std,
          centroid_sim: result.centroid_sim,
          subcenter_sim: result.subcenter_sim,
        },
        pairwise_scores: result.pairwise_scores || [],
        artifacts: {
          grad_cam_report: toStaticUrl(gradCamAbs) || gradCamAbs,
          feature_grid: toStaticUrl(featureGridAbs) || featureGridAbs,
          channel_images: channelAbs.map((p) => toStaticUrl(p) || p),
          verification_json: verifyPayload.results_file,
        },
      };

      setProgress(jobId, "completed", 100, "Pipeline complete");
      res.json(responsePayload);
    } catch (err) {
      setProgress(jobId, "failed", 100, err.message);
      res.status(400).json({
        jobId,
        error: err.message,
      });
    } finally {
      const uploaded = [...vaultFiles, ...questionedFiles];
      for (const file of uploaded) {
        if (file?.path && fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }
      }
      if (fs.existsSync(jobDir)) {
        fs.rmSync(jobDir, { recursive: true, force: true });
      }
    }
  }
);

app.listen(PORT, () => {
  console.log(`SignatureVault API wrapper running at http://localhost:${PORT}`);
  console.log(`Serving generated artifacts at http://localhost:${PORT}/static`);
});
