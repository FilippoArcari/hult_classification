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

  // Serve a poter revocare l'object URL precedente senza dipendere da stale closures.
  const previewUrlRef = useRef<string | null>(null);
  useEffect(() => {
    previewUrlRef.current = previewUrl;
  }, [previewUrl]);

  // Nessuna immagine viene mai salvata: l'object URL vive solo lato client,
  // in memoria, e viene revocato appena non serve più.
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] ?? null;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPredictions(null);
    setError(null);

    if (!selected) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }

  async function handleSubmit() {
    if (!file) return;

    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Passa attraverso il rewrite di next.config.js (/api/* -> FastAPI),
      // niente CORS da gestire e nessun URL del backend hardcoded qui.
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(
          res.status === 413
            ? "Immagine troppo grande."
            : "Il modello non è riuscito ad analizzare l'immagine."
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
          : "Errore di connessione al servizio di analisi."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-xl flex-col gap-6 px-4 py-10">
      <div>
        <h1 className="text-2xl font-semibold text-base-content">
          Analisi lesione cutanea
        </h1>
        <p className="mt-1 text-sm text-base-content/60">
          Carica una foto della lesione. L&apos;immagine viene analizzata e
          scartata subito dopo: non viene salvata da nessuna parte.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input file-input-bordered w-full"
        />

        {previewUrl && (
          <img
            src={previewUrl}
            alt="Anteprima dell'immagine caricata"
            className="max-h-80 w-full rounded-md object-contain bg-base-200"
          />
        )}

        <button
          onClick={handleSubmit}
          disabled={!file || loading}
          className="btn btn-primary self-start"
        >
          {loading && <span className="loading loading-spinner loading-sm" />}
          {loading ? "Analisi in corso" : "Analizza"}
        </button>
      </div>

      {error && (
        <div role="alert" className="alert alert-error">
          <span>{error}</span>
        </div>
      )}

      {predictions && (
        <div className="flex flex-col gap-3">
          <h2 className="text-sm font-medium text-base-content/70">
            Probabilità per classe
          </h2>
          <ul className="flex flex-col gap-2">
            {predictions.map((p) => (
              <li key={p.label} className="flex flex-col gap-1">
                <div className="flex items-baseline justify-between text-sm">
                  <span className="capitalize">{p.label}</span>
                  <span className="tabular-nums text-base-content/60">
                    {(p.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <progress
                  className="progress progress-primary w-full"
                  value={p.probability * 100}
                  max={100}
                />
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="text-xs text-base-content/40">
        Output generato da un modello sperimentale, a solo scopo dimostrativo.
        Non costituisce una diagnosi medica.
      </p>
    </div>
  );
}
