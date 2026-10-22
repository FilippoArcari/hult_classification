"use client";

import { useEffect, useRef, useState } from "react";

interface Prediction {
  label: string;
  probability: number; // 0..1
}

interface PredictResponse {
  predictions: Prediction[];
}

export default function LesionClassifier() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Keep the previous object URL available for cleanup without stale closures.
  const previewUrlRef = useRef<string | null>(null);
  useEffect(() => {
    previewUrlRef.current = previewUrl;
  }, [previewUrl]);

  // Images are never stored: the object URL lives only in browser memory and
  // is revoked as soon as it is no longer needed.
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  function loadFile(selected: File | null) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPredictions(null);
    setError(null);

    if (!selected) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    if (!selected.type.startsWith("image/")) {
      setError("The selected file is not an image.");
      return;
    }

    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    loadFile(e.target.files?.[0] ?? null);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    loadFile(e.dataTransfer.files?.[0] ?? null);
  }

  async function handleSubmit() {
    if (!file) return;

    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // The Next.js rewrite proxies /api/* to FastAPI, so no CORS setup or
      // hardcoded backend URL is needed here.
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(
          res.status === 413
            ? "The image is too large."
            : "The model could not analyze the image."
        );
      }

      const data: PredictResponse = await res.json();
      const sorted = [...data.predictions].sort(
        (a, b) => b.probability - a.probability
      );
      setPredictions(sorted);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not connect to the analysis service."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#F7F8F7] text-[#14181B]">
      <style jsx global>{`
        @import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap");
        .font-ui {
          font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
        }
        .font-data {
          font-family: "IBM Plex Mono", ui-monospace, monospace;
        }
        @keyframes grow-bar {
          from {
            width: 0%;
          }
        }
        .bar-fill {
          animation: grow-bar 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        @media (prefers-reduced-motion: reduce) {
          .bar-fill {
            animation: none;
          }
        }
      `}</style>

      <div className="mx-auto flex max-w-lg flex-col gap-8 px-5 py-14 font-ui">
        {/* Header */}
        <header className="border-b border-[#E2E5E4] pb-6">
          <p className="font-data text-xs tracking-tight text-[#0F6E63]">
            hult · skin lesion analysis
          </p>
          <h1 className="mt-2 text-[1.75rem] font-medium leading-tight text-[#14181B]">
            Upload a photo for analysis
          </h1>
          <p className="mt-2 max-w-[42ch] text-sm leading-relaxed text-[#5B6470]">
            Your image is analyzed in memory and discarded immediately. It is
            never saved.
          </p>
        </header>

        {/* Upload area */}
        <section className="flex flex-col gap-4">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-sm border border-dashed px-6 py-10 text-center transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#0F6E63] ${
              isDragging
                ? "border-[#0F6E63] bg-[#E4F0EE]"
                : "border-[#C7CCCA] bg-white hover:border-[#0F6E63]"
            }`}
          >
            <p className="text-sm text-[#14181B]">
              {file ? file.name : "Drag an image here, or click to choose a file"}
            </p>
            <p className="font-data text-xs text-[#5B6470]">JPG, PNG · max ~10&nbsp;MB</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="sr-only"
            />
          </div>

          {previewUrl && (
            <img
              src={previewUrl}
              alt="Preview of the uploaded image"
              className="max-h-72 w-full rounded-sm border border-[#E2E5E4] bg-white object-contain"
            />
          )}

          <button
            onClick={handleSubmit}
            disabled={!file || loading}
            className="inline-flex w-fit items-center gap-2 rounded-sm bg-[#0F6E63] px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#0C5A51] disabled:cursor-not-allowed disabled:bg-[#C7CCCA] disabled:text-[#5B6470]"
          >
            {loading && (
              <span
                className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white"
                aria-hidden="true"
              />
            )}
            {loading ? "Analyzing image" : "Analyze image"}
          </button>
        </section>

        {error && (
          <div
            role="alert"
            className="border-l-2 border-[#B3261E] bg-[#FBE9E7] px-4 py-3 text-sm text-[#7A1E17]"
          >
            {error}
          </div>
        )}

        {predictions && (
          <section className="flex flex-col gap-4 border-t border-[#E2E5E4] pt-6">
            <h2 className="font-data text-xs uppercase tracking-wide text-[#5B6470]">
              Probability by class
            </h2>
            <ol className="flex flex-col gap-4">
              {predictions.map((p, i) => (
                <li key={p.label} className="flex flex-col gap-1.5">
                  <div className="flex items-baseline justify-between gap-3 text-sm">
                    <span
                      className={`capitalize ${
                        i === 0 ? "font-medium text-[#14181B]" : "text-[#5B6470]"
                      }`}
                    >
                      {p.label}
                    </span>
                    <span className="font-data tabular-nums text-[#14181B]">
                      {(p.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div
                    className="h-1.5 w-full overflow-hidden rounded-full bg-[#E2E5E4]"
                    role="progressbar"
                    aria-valuenow={Math.round(p.probability * 100)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={p.label}
                  >
                    <div
                      className={`bar-fill h-full rounded-full ${
                        i === 0 ? "bg-[#0F6E63]" : "bg-[#8FB6B0]"
                      }`}
                      style={{ width: `${p.probability * 100}%` }}
                    />
                  </div>
                </li>
              ))}
            </ol>
          </section>
        )}

        <p className="border-t border-[#E2E5E4] pt-4 text-xs leading-relaxed text-[#5B6470]">
          Output from an experimental model, for demonstration purposes only.
          This is not a medical diagnosis.
        </p>
      </div>
    </div>
  );
}
