import { createLucideIcon } from "lucide-react";

export const SplitCenterIcon = createLucideIcon("SplitCenterIcon", [
  ["path", { d: "M5 4h4v16H5", key: "left-bracket" }],
  ["path", { d: "M19 4h-4v16h4", key: "right-bracket" }],
]);

export const SplitLeftDotsIcon = createLucideIcon("SplitLeftDotsIcon", [
  ["path", { d: "M20 4h-4v16h4", key: "bracket" }],

  ["path", { d: "M6 4h.01", key: "d1" }],
  ["path", { d: "M6 20h.01", key: "d3" }],

  ["path", { d: "M11 4h.01", key: "d4" }],
  ["path", { d: "M11 8h.01", key: "d5" }],
  ["path", { d: "M11 12h.01", key: "d6" }],
  ["path", { d: "M11 16h.01", key: "d7" }],
  ["path", { d: "M11 20h.01", key: "d8" }],
]);

export const SplitRightDotsIcon = createLucideIcon("SplitRightDotsIcon", [
  ["path", { d: "M4 4h4v16H4", key: "bracket" }],

  ["path", { d: "M13 4h.01", key: "d1" }],
  ["path", { d: "M13 8h.01", key: "d2" }],
  ["path", { d: "M13 12h.01", key: "d3" }],
  ["path", { d: "M13 16h.01", key: "d4" }],
  ["path", { d: "M13 20h.01", key: "d5" }],

  ["path", { d: "M18 4h.01", key: "d6" }],
  ["path", { d: "M18 20h.01", key: "d8" }],
]);
