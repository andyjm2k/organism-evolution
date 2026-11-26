/** Unit tests for the web Camera zoom. */

import { describe, expect, it } from "vitest";
import { Camera } from "../src/render/camera";

describe("Camera zoom", () => {
  it("keeps the world point under the cursor fixed", () => {
    const camera = new Camera(4000, 4000, 800, 600);
    camera.centerOn(2000, 2000, false);
    const before = camera.screenToWorld(400, 300);
    camera.zoomAt(400, 300, 1);
    expect(camera.zoom).toBeGreaterThan(1);
    const after = camera.screenToWorld(400, 300);
    expect(after[0]).toBeCloseTo(before[0], 3);
    expect(after[1]).toBeCloseTo(before[1], 3);
  });

  it("clamps to min/max zoom", () => {
    const camera = new Camera(4000, 4000, 800, 600);
    for (let i = 0; i < 30; i += 1) camera.zoomAt(400, 300, 1);
    expect(camera.zoom).toBeCloseTo(camera.maxZoom, 6);
    for (let i = 0; i < 60; i += 1) camera.zoomAt(400, 300, -1);
    expect(camera.zoom).toBeCloseTo(camera.minZoom, 6);
  });

  it("scales world-to-screen and shrinks the viewport rect", () => {
    const camera = new Camera(4000, 4000, 800, 600);
    camera.zoomAt(0, 0, 1);
    const [sx, sy] = camera.worldToScreen(100, 100);
    expect(sx).toBeCloseTo(100 * camera.zoom, 3);
    expect(sy).toBeCloseTo(100 * camera.zoom, 3);
    const [wx, wy] = camera.screenToWorld(sx, sy);
    expect(wx).toBeCloseTo(100, 3);
    expect(wy).toBeCloseTo(100, 3);
    const [x0, y0, x1, y1] = camera.viewportRect();
    expect(x1 - x0).toBeCloseTo(800 / camera.zoom, 3);
    expect(y1 - y0).toBeCloseTo(600 / camera.zoom, 3);
  });

  it("scales drag distance by zoom", () => {
    const camera = new Camera(4000, 4000, 800, 600);
    camera.centerOn(2000, 2000, false);
    camera.zoomAt(400, 300, 1);
    const startX = camera.offsetX;
    camera.beginDrag(400, 300);
    camera.dragTo(500, 300);
    expect(camera.offsetX).toBeCloseTo(startX - 100 / camera.zoom, 3);
  });
});
