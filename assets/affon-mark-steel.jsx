import { useState } from "react";

const STEEL = "#94A3B8";
const INK = "#080C14";
const SLATE = "#1A2438";
const MIST = "#7A8BA0";
const PAPER = "#F0F4F8";
const DIM = "#253048";
const AMBER = "#E8A038";

const NULL_R = 6;
const NULL_CY = 104;
const ARC_AMP = 14;
const ARC_PEAK = 87;

function Mark({ size = 140 }) {
  const fs = 28;
  const cols = [36, 70, 104];
  const d = NULL_R * 0.72;
  const arcPath = `M 60 58 Q ${ARC_PEAK} ${58 - ARC_AMP} 116 58`;

  return (
    <svg width={size} height={size} viewBox="0 0 140 140" fill="none">
      {/* brackets */}
      <path d="M18 24 L10 24 L10 116 L18 116" stroke={STEEL} strokeWidth={3} fill="none" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M122 24 L130 24 L130 116 L122 116" stroke={STEEL} strokeWidth={3} fill="none" strokeLinecap="round" strokeLinejoin="round"/>

      {/* row 1: a  f  f */}
      <text x={cols[0]} y="72" fill={STEEL} fontSize={fs} fontFamily="Georgia, serif" textAnchor="middle">a</text>
      <text x={cols[1]} y="72" fill={STEEL} fontSize={fs} fontFamily="Georgia, serif" textAnchor="middle" opacity="0.85">f</text>
      <text x={cols[2]} y="72" fill={STEEL} fontSize={fs} fontFamily="Georgia, serif" textAnchor="middle" opacity="0.65">f</text>

      {/* arc crossbar */}
      <path d={arcPath} stroke={STEEL} strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.35"/>
      {/* amber node at arc peak */}
      <circle cx={ARC_PEAK} cy={58 - ARC_AMP} r="2.5" fill={AMBER} opacity="0.95"/>

      {/* row 2: ∅  o  n */}
      <circle cx={cols[0]} cy={NULL_CY} r={NULL_R} stroke={STEEL} strokeWidth="1.6" fill="none" opacity="0.4"/>
      <line x1={cols[0]-d} y1={NULL_CY+d} x2={cols[0]+d} y2={NULL_CY-d} stroke={STEEL} strokeWidth="1.6" strokeLinecap="round" opacity="0.4"/>
      <text x={cols[1]} y="112" fill={STEEL} fontSize={fs} fontFamily="Georgia, serif" textAnchor="middle" opacity="0.75">o</text>
      <text x={cols[2]} y="112" fill={STEEL} fontSize={fs} fontFamily="Georgia, serif" textAnchor="middle" opacity="0.85">n</text>

      {/* hairlines */}
      <line x1="14" y1="82" x2="126" y2="82" stroke={STEEL} strokeWidth="0.75" opacity="0.1"/>
      <line x1="53" y1="28" x2="53" y2="112" stroke={STEEL} strokeWidth="0.75" opacity="0.1"/>
      <line x1="87" y1="28" x2="87" y2="112" stroke={STEEL} strokeWidth="0.75" opacity="0.1"/>
    </svg>
  );
}

export default function AffonFinal() {
  return (
    <div style={{ background: INK, minHeight: "100vh", padding: "56px 40px", fontFamily: "system-ui, sans-serif", color: PAPER }}>

      {/* Header */}
      <div style={{ marginBottom: 56, borderBottom: `1px solid ${SLATE}`, paddingBottom: 28 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 8 }}>
          <span style={{ fontFamily: "Georgia, serif", fontSize: 42, letterSpacing: "-1.5px" }}>affon</span>
          <span style={{ background: SLATE, color: STEEL, fontSize: 11, fontFamily: "monospace", padding: "3px 8px", borderRadius: 4 }}>
            steel · arc · amp=14 peak=87 · r=6 cy=104
          </span>
        </div>
      </div>

      {/* Lockups */}
      <div style={{ marginBottom: 48 }}>
        <div style={{ fontSize: 10, color: MIST, marginBottom: 12, fontFamily: "monospace" }}>LOCKUP</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 20 }}>
          <div>
            <div style={{ fontSize: 10, color: MIST, marginBottom: 8, fontFamily: "monospace" }}>DARK</div>
            <div style={{ display: "flex", alignItems: "center", gap: 18, background: INK, border: `1px solid ${SLATE}`, padding: "24px 32px", borderRadius: 12 }}>
              <Mark size={140}/>
              <span style={{ fontFamily: "Georgia, serif", fontSize: 40, color: PAPER, letterSpacing: "-1.5px", fontWeight: 400 }}>affon</span>
            </div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: MIST, marginBottom: 8, fontFamily: "monospace" }}>LIGHT</div>
            <div style={{ display: "flex", alignItems: "center", gap: 18, background: PAPER, padding: "24px 32px", borderRadius: 12 }}>
              <Mark size={140}/>
              <span style={{ fontFamily: "Georgia, serif", fontSize: 40, color: INK, letterSpacing: "-1.5px", fontWeight: 400 }}>affon</span>
            </div>
          </div>
        </div>
      </div>

      {/* Mark at sizes */}
      <div style={{ marginBottom: 48 }}>
        <div style={{ fontSize: 10, color: MIST, marginBottom: 12, fontFamily: "monospace" }}>MARK ONLY</div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 28, background: SLATE, padding: "28px 32px", borderRadius: 12 }}>
          {[140, 96, 64, 40, 24].map(sz => (
            <div key={sz} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
              <Mark size={sz}/>
              <span style={{ fontSize: 9, color: MIST, fontFamily: "monospace" }}>{sz}px</span>
            </div>
          ))}
        </div>
      </div>

      {/* npm badge */}
      <div style={{ marginBottom: 48 }}>
        <div style={{ fontSize: 10, color: MIST, marginBottom: 12, fontFamily: "monospace" }}>NPM BADGE</div>
        <div style={{ display: "inline-flex", alignItems: "center" }}>
          <div style={{ background: SLATE, padding: "5px 12px 5px 8px", display: "flex", alignItems: "center", gap: 8, borderRadius: "6px 0 0 6px" }}>
            <Mark size={28}/>
            <span style={{ fontFamily: "Georgia, serif", fontSize: 14, color: PAPER, letterSpacing: "-0.5px" }}>affon</span>
          </div>
          <div style={{ background: AMBER, padding: "5px 12px", borderRadius: "0 6px 6px 0" }}>
            <span style={{ fontFamily: "monospace", fontSize: 12, color: INK, fontWeight: 700 }}>v0.1.0</span>
          </div>
        </div>
      </div>

      {/* Favicon grid */}
      <div>
        <div style={{ fontSize: 10, color: MIST, marginBottom: 12, fontFamily: "monospace" }}>FAVICON</div>
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
          {[64, 48, 32, 20].map(sz => (
            <div key={sz} style={{
              background: INK, border: `1px solid ${SLATE}`,
              borderRadius: sz > 40 ? 12 : sz > 28 ? 8 : 6,
              width: sz + 16, height: sz + 16,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <Mark size={sz}/>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
