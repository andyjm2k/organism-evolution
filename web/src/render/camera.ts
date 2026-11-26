/** Viewport into the 4000×4000 world. */

export class Camera {
  offsetX = 0;
  offsetY = 0;
  /** Zoom factor (1.0 = one world unit per screen pixel). */
  zoom = 1;
  readonly minZoom = 0.5;
  readonly maxZoom = 3;
  private track: [number, number] | null = null;
  private dragging = false;
  private dragStart: [number, number] | null = null;
  private dragOrigin: [number, number] | null = null;

  constructor(
    readonly worldWidth: number,
    readonly worldHeight: number,
    public viewWidth: number,
    public viewHeight: number,
    readonly smoothing = 0.15,
  ) {}

  clamp(): void {
    const maxX = Math.max(0, this.worldWidth - this.viewWidth / this.zoom);
    const maxY = Math.max(0, this.worldHeight - this.viewHeight / this.zoom);
    this.offsetX = Math.max(0, Math.min(this.offsetX, maxX));
    this.offsetY = Math.max(0, Math.min(this.offsetY, maxY));
  }

  /** Zoom toward the cursor, keeping the world point under it fixed. */
  zoomAt(x: number, y: number, delta: number): void {
    const factor = delta > 0 ? 1.1 : 1 / 1.1;
    const next = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom * factor));
    if (next === this.zoom) {
      return;
    }
    const [worldX, worldY] = this.screenToWorld(x, y);
    this.zoom = next;
    this.offsetX = worldX - x / next;
    this.offsetY = worldY - y / next;
    this.clamp();
  }

  centerOn(x: number, y: number, smooth = false): void {
    const targetX = x - this.viewWidth / this.zoom / 2;
    const targetY = y - this.viewHeight / this.zoom / 2;
    if (smooth && this.smoothing > 0) {
      this.offsetX += (targetX - this.offsetX) * this.smoothing;
      this.offsetY += (targetY - this.offsetY) * this.smoothing;
    } else {
      this.offsetX = targetX;
      this.offsetY = targetY;
    }
    this.clamp();
  }

  setTrack(position: [number, number] | null): void {
    this.track = position;
  }

  updateTracking(): void {
    if (this.track) {
      this.centerOn(this.track[0], this.track[1], true);
    }
  }

  worldToScreen(x: number, y: number): [number, number] {
    return [(x - this.offsetX) * this.zoom, (y - this.offsetY) * this.zoom];
  }

  screenToWorld(x: number, y: number): [number, number] {
    return [x / this.zoom + this.offsetX, y / this.zoom + this.offsetY];
  }

  viewportRect(): [number, number, number, number] {
    return [this.offsetX, this.offsetY, this.offsetX + this.viewWidth / this.zoom, this.offsetY + this.viewHeight / this.zoom];
  }

  beginDrag(x: number, y: number): void {
    this.dragging = true;
    this.dragStart = [x, y];
    this.dragOrigin = [this.offsetX, this.offsetY];
    this.track = null;
  }

  dragTo(x: number, y: number): void {
    if (!this.dragging || !this.dragStart || !this.dragOrigin) {
      return;
    }
    // A screen drag moves the world by the zoomed distance.
    this.offsetX = this.dragOrigin[0] - (x - this.dragStart[0]) / this.zoom;
    this.offsetY = this.dragOrigin[1] - (y - this.dragStart[1]) / this.zoom;
    this.clamp();
  }

  endDrag(): void {
    this.dragging = false;
    this.dragStart = null;
    this.dragOrigin = null;
  }

  isDragging(): boolean {
    return this.dragging;
  }
}
