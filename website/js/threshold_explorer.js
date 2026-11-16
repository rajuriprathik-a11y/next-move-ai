/**
 * Next Move AI — Interactive Risk Threshold Explorer
 * Explores classification metrics across thresholds using 219,983 out-of-fold XGBoost validation predictions.
 */

const ThresholdExplorer = {
    oofData: null,
    currentThreshold: 0.2880,

    async init() {
        try {
            const resp = await fetch('data/oof_threshold_metrics.json');
            this.oofData = await resp.json();
            this.bindEvents();
            this.updateDisplay(this.currentThreshold);
        } catch (err) {
            console.error('Failed to load OOF threshold data:', err);
        }
    },

    bindEvents() {
        const slider = document.getElementById('threshold-slider');
        const numInput = document.getElementById('threshold-number');
        const resetBtn = document.getElementById('threshold-reset-btn');

        if (slider) {
            slider.addEventListener('input', (e) => {
                let val = parseFloat(e.target.value);
                // Magnetic snap to official operating point if user drags nearby
                if (Math.abs(val - 0.2880) <= 0.006) {
                    val = 0.2880;
                }
                this.updateDisplay(val);
                if (numInput) numInput.value = val.toFixed(4);
            });
        }

        if (numInput) {
            numInput.addEventListener('change', (e) => {
                let val = parseFloat(e.target.value);
                if (isNaN(val)) val = 0.2880;
                val = Math.max(0.05, Math.min(0.85, val));
                numInput.value = val.toFixed(4);
                if (slider) slider.value = val;
                this.updateDisplay(val);
            });
        }

        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                const val = 0.2880;
                if (slider) slider.value = val;
                if (numInput) numInput.value = val.toFixed(4);
                this.updateDisplay(val);
            });
        }
    },

    getMetricsForThreshold(targetTau) {
        if (!this.oofData || !this.oofData.threshold_sweep) return null;

        const sweep = this.oofData.threshold_sweep;
        // Exact match check first
        for (let i = 0; i < sweep.length; i++) {
            if (Math.abs(sweep[i].threshold - targetTau) < 1e-4) {
                return sweep[i];
            }
        }

        // Closest match fallback (exact stored sweep point, strictly no interpolation)
        let closest = sweep[0];
        let minDiff = Math.abs(closest.threshold - targetTau);

        for (let i = 1; i < sweep.length; i++) {
            const diff = Math.abs(sweep[i].threshold - targetTau);
            if (diff < minDiff) {
                minDiff = diff;
                closest = sweep[i];
            }
        }
        return closest;
    },

    updateDisplay(tau) {
        this.currentThreshold = tau;
        const metrics = this.getMetricsForThreshold(tau);
        if (!metrics) return;

        // Threshold badges
        const tauDisplay = document.getElementById('current-tau-display');
        const opBadge = document.getElementById('operating-point-badge');
        if (tauDisplay) tauDisplay.textContent = metrics.threshold.toFixed(4);

        const isOperatingPoint = Math.abs(metrics.threshold - 0.2880) < 0.001;
        if (opBadge) {
            opBadge.style.display = isOperatingPoint ? 'inline-flex' : 'none';
        }

        // Key metrics
        this.setText('metric-recall', `${metrics.recall.toFixed(2)}%`);
        this.setText('metric-precision', `${metrics.precision.toFixed(2)}%`);
        this.setText('metric-specificity', `${metrics.specificity.toFixed(2)}%`);
        this.setText('metric-f1', `${metrics.f1_score.toFixed(2)}%`);
        this.setText('metric-fpr', `${metrics.fpr.toFixed(2)}%`);

        // Volume flagged
        this.setText('metric-flagged', `${metrics.flagged.toLocaleString()} / 219,983 (${metrics.flagged_pct.toFixed(2)}%)`);

        // Confusion Matrix
        this.setText('cm-tp', metrics.tp.toLocaleString());
        this.setText('cm-fn', metrics.fn.toLocaleString());
        this.setText('cm-fp', metrics.fp.toLocaleString());
        this.setText('cm-tn', metrics.tn.toLocaleString());

        // Percentages in CM
        const totalRisk = metrics.tp + metrics.fn;
        const totalCorrect = metrics.fp + metrics.tn;
        this.setText('cm-tp-pct', `(${((metrics.tp / totalRisk) * 100).toFixed(1)}% of failures)`);
        this.setText('cm-fn-pct', `(${((metrics.fn / totalRisk) * 100).toFixed(1)}% missed)`);
        this.setText('cm-fp-pct', `(${((metrics.fp / totalCorrect) * 100).toFixed(1)}% false alarms)`);
        this.setText('cm-tn-pct', `(${((metrics.tn / totalCorrect) * 100).toFixed(1)}% cleared)`);

        // Update trade-off note
        const noteEl = document.getElementById('threshold-tradeoff-note');
        if (noteEl) {
            if (metrics.threshold < 0.20) {
                noteEl.innerHTML = `<strong>Ultra-High Recall Mode:</strong> Captures over ${metrics.recall.toFixed(0)}% of struggling interactions, but flags ${metrics.flagged_pct.toFixed(1)}% of all questions, creating substantial false alarms.`;
            } else if (isOperatingPoint) {
                noteEl.innerHTML = `<strong>Selected Operating Point (\\(\\tau = 0.2880\\)):</strong> Balanced for proactive educational assistance (~75% recall, ~49% precision), capturing 3 out of 4 struggling attempts with manageable false alarm overhead.`;
            } else if (metrics.threshold > 0.45) {
                noteEl.innerHTML = `<strong>High Precision Mode:</strong> Precision rises to ${metrics.precision.toFixed(1)}%, but sensitivity drops to ${metrics.recall.toFixed(1)}%, missing ${(100 - metrics.recall).toFixed(1)}% of student failures. Best suited for high-stakes interventions.`;
            } else {
                noteEl.innerHTML = `Operating at threshold \\(\\tau = ${metrics.threshold.toFixed(4)}\\). Yields ${metrics.recall.toFixed(1)}% recall with ${metrics.precision.toFixed(1)}% precision on validation data.`;
            }
            if (window.renderMath) window.renderMath(noteEl);
        }
    },

    setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }
};

window.ThresholdExplorer = ThresholdExplorer;
document.addEventListener('DOMContentLoaded', () => ThresholdExplorer.init());
