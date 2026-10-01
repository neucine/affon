import fs from "std:fs";
import { getEnv } from "std:process";

const path = getEnv("AFFON_WIKITEXT_CANARY_SUMMARY");
if (!path) throw new AffonError("invalid_arg", "Set AFFON_WIKITEXT_CANARY_SUMMARY");
const summary = JSON.parse(fs.readFileSync(path));
const proof = summary.candidateProof;
if (summary.format !== "affon-decoder-program-canary/v1" ||
    !proof || !["cpu", "metal", "cuda"].includes(proof.backend) ||
    typeof proof.loss !== "number" || !Number.isFinite(proof.loss) || proof.loss <= 0 ||
    !Array.isArray(proof.resumedParameter) || proof.resumedParameter.length === 0 ||
    proof.resumedParameter.some((value: unknown) => typeof value !== "number" || !Number.isFinite(value)) ||
    typeof proof.explanation !== "string" || !proof.explanation.includes("authored_program") ||
    summary.state?.checkpoint_rebound !== true)
  throw new AffonError("assertion", "decoder Program canary evidence is incomplete");
console.log(JSON.stringify({ passed: true, backend: proof.backend, loss: proof.loss, duration_ns: summary.duration_ns }));
