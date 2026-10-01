import fs from "std:fs";
import { getEnv } from "std:process";
import { runDecoderProof, type CandidateBackend } from "../../src/js/compute/candidate.ts";

const backend = (getEnv("AFFON_T13_CANDIDATE_BACKEND") ?? getEnv("AFFON_TRAIN_DEVICE") ?? "cpu") as CandidateBackend;
if (!(["cpu", "metal", "cuda"] as string[]).includes(backend))
  throw new AffonError("invalid_arg", "decoder backend must be cpu, metal, or cuda");

const started = Date.now();
const candidateProof = await runDecoderProof(backend);
const duration_ns = Math.max(1, (Date.now() - started) * 1_000_000);
const summary = {
  format: "affon-decoder-program-canary/v1",
  backend,
  duration_ns,
  candidateProof,
  state: { parameter_count: candidateProof.resumedParameter.length, checkpoint_rebound: true },
};
const explanationPath = getEnv("AFFON_T13_EXPLANATION_PATH");
if (explanationPath) fs.writeFileSync(explanationPath, candidateProof.explanation);
const summaryPath = getEnv("AFFON_TRAIN_SUMMARY_PATH");
if (summaryPath) fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary));
