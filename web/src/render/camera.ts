/** Viewport into the 4000×4000 world. */

export class Camera {
  offsetX = 0;
  offsetY = 0;
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
    const maxX = Math.max(0, this.worldWidth - this.viewWidth);
    const maxY = Math.max(0, this.worldHeight - this.viewHeight);
    this.offsetX = Math.max(0, Math.min(this.offsetX, maxX));
    this.offsetY = Math.max(0, Math.min(this.offsetY, maxY));
  }

  centerOn(x: number, y: number, smooth = false): void {
    const targetX = x - this.viewWidth / 2;
    const targetY = y - this.viewHeight / 2;
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
    return [x - this.offsetX, y - this.offsetY];
  }

  screenToWorld(x: number, y: number): [number, number] {
    return [x + this.offsetX, y + this.offsetY];
  }

  viewportRect(): [number, number, number, number] {
    return [this.offsetX, this.offsetY, this.offsetX + this.viewWidth, this.offsetY + this.viewHeight];
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
    this.offsetX = this.dragOrigin[0] - (x - this.dragStart[0]);
    this.offsetY = this.dragOrigin[1] - (y - this.dragStart[1]);
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
