// js/particles.js -- Stormlight particle system and gemstone path generator

export class StormParticles {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.particles = [];
    this.running = true;
    this.resize();
    window.addEventListener('resize', () => this.resize());
    document.addEventListener('visibilitychange', () => {
      this.running = !document.hidden;
      if (this.running) this.loop();
    });
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  init(count) {
    this.particles = [];
    for (let i = 0; i < count; i++) {
      this.particles.push(this.spawn(true));
    }
    this.loop();
  }

  spawn(randomY) {
    const layer = Math.random() > 0.5 ? 0 : 1;
    return {
      x: Math.random() * this.canvas.width,
      y: randomY ? Math.random() * this.canvas.height : this.canvas.height + 20,
      vy: -(0.12 + Math.random() * 0.28) * (layer === 0 ? 0.5 : 1),
      wobbleSpeed: 0.0008 + Math.random() * 0.002,
      wobbleAmp: 12 + Math.random() * 25,
      phase: Math.random() * Math.PI * 2,
      size: layer === 0 ? (2.5 + Math.random() * 3) : (1 + Math.random() * 2),
      opacity: layer === 0 ? (0.035 + Math.random() * 0.05) : (0.05 + Math.random() * 0.09),
      time: Math.random() * 10000,
    };
  }

  loop() {
    if (!this.running) return;
    const { ctx, canvas, particles } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const p of particles) {
      p.time += 1;
      p.y += p.vy;
      const wx = Math.sin(p.time * p.wobbleSpeed + p.phase) * p.wobbleAmp;
      if (p.y < -30) {
        p.y = canvas.height + 30;
        p.x = Math.random() * canvas.width;
        p.phase = Math.random() * Math.PI * 2;
      }
      ctx.beginPath();
      ctx.arc(p.x + wx, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(200,223,255,' + p.opacity + ')';
      ctx.fill();
    }
    requestAnimationFrame(() => this.loop());
  }
}

export function gemPath(r, facets) {
  const n = facets || 6;
  const offset = -Math.PI / 2;
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI / n) * i + offset;
    pts.push((r * Math.cos(a)).toFixed(2) + ',' + (r * Math.sin(a)).toFixed(2));
  }
  return 'M' + pts.join('L') + 'Z';
}
