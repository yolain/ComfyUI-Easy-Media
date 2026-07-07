import { createLucideIcon } from "lucide-react";

export const SplitCenterIcon = createLucideIcon("SplitCenterIcon", [
  ["path", { d: "M7 5H4v14h3", key: "left-bracket" }],
  ["path", { d: "M17 5h3v14h-3", key: "right-bracket" }],
]);

export const SplitLeftDotsIcon = createLucideIcon("SplitLeftDotsIcon", [
  ["path", { d: "M16 5h4v14h-4", key: "bracket" }],

  ["path", { d: "M8 5h.01", key: "d1" }],
  ["path", { d: "M8 12h.01", key: "d2" }],
  ["path", { d: "M8 19h.01", key: "d3" }],

  ["path", { d: "M12 5h.01", key: "d4" }],
  ["path", { d: "M12 8.5h.01", key: "d5" }],
  ["path", { d: "M12 12h.01", key: "d6" }],
  ["path", { d: "M12 15.5h.01", key: "d7" }],
  ["path", { d: "M12 19h.01", key: "d8" }],
]);

export const SplitRightDotsIcon = createLucideIcon("SplitRightDotsIcon", [
  ["path", { d: "M4 5h4v14H4", key: "bracket" }],

  ["path", { d: "M12 5h.01", key: "d1" }],
  ["path", { d: "M12 8.5h.01", key: "d2" }],
  ["path", { d: "M12 12h.01", key: "d3" }],
  ["path", { d: "M12 15.5h.01", key: "d4" }],
  ["path", { d: "M12 19h.01", key: "d5" }],

  ["path", { d: "M16 5h.01", key: "d6" }],
  ["path", { d: "M16 12h.01", key: "d7" }],
  ["path", { d: "M16 19h.01", key: "d8" }],
]);