import fs from "std:fs";
import { getEnv, run } from "std:process";
import { semantic_loss_report } from "../../../packages/@affon/onnx/src/index.ts";

type Comparison = {
  passed: boolean;
  mismatches: number;
  max_absolute_error: number;
};
type CandidateResult = {
  case: number;
  onnx: Comparison;
  pytorch: Comparison;
  top1: number;
  values: number[];
  explanation: string;
  passed: boolean;
};

const directory = getEnv("AFFON_ONNX_DIR");
const executable = getEnv("AFFON_CANDIDATE_BIN");
const reportPath = getEnv("AFFON_HF_REPORT");
if (!directory || !executable || !reportPath)
  throw Error("Set AFFON_ONNX_DIR, AFFON_CANDIDATE_BIN and AFFON_HF_REPORT");

const results: CandidateResult[] = [];
for (let caseIndex = 0; caseIndex < 2; caseIndex++) {
  const result = await run({
    cmd: executable,
    args: ["onnx", directory, `${caseIndex}`],
    check: false,
    maxOutputBytes: 2 * 1024 * 1024,
  });
  if (result.exitCode !== 0)
    throw Error(`candidate Program execution failed: ${result.stderr.slice(0, 500)}`);
  const parsed = JSON.parse(result.stdout) as CandidateResult;
  if (parsed.case !== caseIndex || parsed.values.length !== 1001)
    throw Error("candidate Program returned an invalid MobileNet result");
  results.push(parsed);
}

const report = {
  format: "affon-candidate-onnx-audit/v1",
  route: "candidate-program",
  device: "cpu",
  semantic_loss: semantic_loss_report(directory),
  results,
  passed: results.every((result) => result.passed),
};
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify({
  ...report,
  results: results.map(({ values: _values, explanation, ...result }) => ({
    ...result,
    explanation_bytes: explanation.length,
  })),
}));
if (!report.passed) throw Error("candidate ONNX parity check failed; see saved report");
