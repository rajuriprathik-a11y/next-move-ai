/**
 * Next Move AI — Live Bayesian Smoothing Calculator
 * Computes A = (C + alpha * P) / (N + alpha) with cold-start visualization.
 */

const SmoothingCalculator = {
    init() {
        this.bindEvents();
        this.recalculate();
    },

    bindEvents() {
        const inputs = ['calc-c', 'calc-n', 'calc-p', 'calc-alpha'];
        inputs.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('input', () => this.recalculate());
            }
        });

        // Quick presets
        document.querySelectorAll('.calc-preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const c = parseFloat(btn.dataset.c);
                const n = parseFloat(btn.dataset.n);
                const p = parseFloat(btn.dataset.p);
                const alpha = parseFloat(btn.dataset.alpha);

                if (!isNaN(c)) document.getElementById('calc-c').value = c;
                if (!isNaN(n)) document.getElementById('calc-n').value = n;
                if (!isNaN(p)) document.getElementById('calc-p').value = p;
                if (!isNaN(alpha)) document.getElementById('calc-alpha').value = alpha;

                this.recalculate();
            });
        });
    },

    recalculate() {
        const cInput = document.getElementById('calc-c');
        const nInput = document.getElementById('calc-n');
        const pInput = document.getElementById('calc-p');
        const alphaInput = document.getElementById('calc-alpha');

        if (!cInput || !nInput || !pInput || !alphaInput) return;

        let c = parseFloat(cInput.value) || 0;
        let n = parseFloat(nInput.value) || 0;
        let p = parseFloat(pInput.value) || 0.6586;
        let alpha = parseFloat(alphaInput.value) || 10;

        // Invariant: c cannot exceed n
        if (c > n) {
            c = n;
            cInput.value = c;
        }

        const smoothed = (c + alpha * p) / (n + alpha);
        const raw = n > 0 ? (c / n) : null;
        const shrinkage = raw !== null ? (smoothed - raw) : (smoothed - p);

        // Update displays
        const resEl = document.getElementById('calc-result-smoothed');
        const rawEl = document.getElementById('calc-result-raw');
        const diffEl = document.getElementById('calc-result-shrinkage');
        const explEl = document.getElementById('calc-explanation');

        if (resEl) resEl.textContent = (smoothed * 100).toFixed(2) + '%';
        if (rawEl) rawEl.textContent = raw !== null ? (raw * 100).toFixed(2) + '%' : 'Undefined (0/0)';
        if (diffEl) {
            const sign = shrinkage >= 0 ? '+' : '';
            diffEl.textContent = `${sign}${(shrinkage * 100).toFixed(2)}% vs raw`;
            diffEl.className = 'calc-stat-sub ' + (shrinkage >= 0 ? 'text-positive' : 'text-negative');
        }

        if (explEl) {
            if (n === 0) {
                explEl.innerHTML = `<strong>Cold-Start Active (\\(N = 0\\)):</strong> The student has zero prior attempts. The smoothed estimate shrinks completely to the historical training prior: \\(A = \\frac{0 + 10 \\times ${p.toFixed(4)}}{0 + 10} = ${p.toFixed(4)}\\).`;
            } else if (n < alpha) {
                explEl.innerHTML = `<strong>Heavy Shrinkage (\\(N = ${n} < \\alpha = ${alpha}\\)):</strong> With few observations, the estimate is weighted more heavily toward the prior (${(p * 100).toFixed(1)}%) than the noisy empirical average (${raw !== null ? (raw * 100).toFixed(1) : 0}%).`;
            } else {
                explEl.innerHTML = `<strong>Data-Dominated (\\(N = ${n} \\ge \\alpha = ${alpha}\\)):</strong> As interaction volume grows, the empirical evidence (${(raw * 100).toFixed(1)}%) gradually overwhelms the prior, converging toward the true observed student rate.`;
            }
            if (window.renderMath) window.renderMath(explEl);
        }
    }
};

window.SmoothingCalculator = SmoothingCalculator;
document.addEventListener('DOMContentLoaded', () => SmoothingCalculator.init());
