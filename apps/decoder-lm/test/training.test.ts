import { test, expect } from "std:test";
import { runDecoderProof } from "../../../src/js/compute/candidate.ts";

test("decoder Program differentiates, updates explicit state, and resumes", async () => {
  const proof = await runDecoderProof("cpu");
  expect(Number.isFinite(proof.loss)).toBe(true);
  expect(proof.loss > 0).toBe(true);
  expect(proof.resumedParameter.length).toBe(12);
  expect(proof.resumedParameter.every(Number.isFinite)).toBe(true);
  expect(proof.explanation.includes("authored_program")).toBe(true);
});
