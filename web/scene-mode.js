export const SCENE_EDGE_TOLERANCE = 2;

export function decideSceneWheel({
  deltaY,
  panelTop,
  panelBottom,
  viewportHeight,
  currentIndex,
  panelCount,
  edgeTolerance = SCENE_EDGE_TOLERANCE,
}) {
  const direction = Math.sign(Number(deltaY) || 0);
  const index = Number.isInteger(currentIndex) ? currentIndex : 0;
  const count = Number.isInteger(panelCount) ? panelCount : 0;
  const tolerance = Math.max(0, Number(edgeTolerance) || 0);

  if (!direction || count < 1) return { action: "native", nextIndex: index };

  if (direction > 0) {
    const hasContentBelow = Number(panelBottom) > Number(viewportHeight) + tolerance;
    if (hasContentBelow || index >= count - 1) {
      return { action: "native", nextIndex: index };
    }
    return { action: "navigate", nextIndex: index + 1 };
  }

  const hasContentAbove = Number(panelTop) < -tolerance;
  if (hasContentAbove || index <= 0) {
    return { action: "native", nextIndex: index };
  }
  return { action: "navigate", nextIndex: index - 1 };
}
