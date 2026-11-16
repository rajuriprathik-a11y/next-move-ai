/**
 * Next Move AI — Data Visualizations (Chart.js)
 * Manages all academic performance charts with dynamic theme adaptation.
 */

const ChartManager = {
    resultsData: null,
    charts: {},

    async init() {
        try {
            const resp = await fetch('data/results.json');
            this.resultsData = await resp.json();

            // Wait for Chart.js to be available
            if (typeof Chart === 'undefined') {
                console.warn('Chart.js not yet loaded, waiting...');
                setTimeout(() => this.init(), 100);
                return;
            }

            this.setupDefaults();
            this.renderAll();

            window.addEventListener('themeChanged', () => {
                this.updateThemeColors();
            });
        } catch (err) {
            console.error('Failed to load results.json for charts:', err);
        }
    },

    getThemeColors() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        return {
            isDark,
            text: isDark ? '#e2e8f0' : '#1e293b',
            muted: isDark ? '#94a3b8' : '#64748b',
            grid: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)',
            primary: '#3b82f6',
            primaryTrans: isDark ? 'rgba(59, 130, 246, 0.25)' : 'rgba(59, 130, 246, 0.15)',
            secondary: '#10b981',
            accent: '#f59e0b',
            danger: '#ef4444',
            purple: '#8b5cf6',
            cardBg: isDark ? '#1e293b' : '#ffffff'
        };
    },

    setupDefaults() {
        const c = this.getThemeColors();
        Chart.defaults.color = c.muted;
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        Chart.defaults.font.size = 12;
        Chart.defaults.plugins.tooltip.padding = 10;
        Chart.defaults.plugins.tooltip.cornerRadius = 8;
    },

    updateThemeColors() {
        const c = this.getThemeColors();
        Chart.defaults.color = c.muted;

        Object.values(this.charts).forEach(chart => {
            if (!chart) return;
            if (chart.options.scales) {
                if (chart.options.scales.x) {
                    chart.options.scales.x.grid.color = c.grid;
                    chart.options.scales.x.ticks.color = c.muted;
                }
                if (chart.options.scales.y) {
                    chart.options.scales.y.grid.color = c.grid;
                    chart.options.scales.y.ticks.color = c.muted;
                }
            }
            chart.update();
        });
    },

    renderAll() {
        this.renderDatasetCharts();
        this.renderSingleGroupChart();
        this.renderAblationChart();
        this.renderModelComparisonChart();
        this.renderTestCurves();
        this.renderFeatureImportance();
    },

    // 1. Dataset Explorer Charts
    renderDatasetCharts() {
        const c = this.getThemeColors();
        const ctxClass = document.getElementById('chart-class-balance');
        if (ctxClass && this.resultsData.dataset) {
            this.charts.classBalance = new Chart(ctxClass, {
                type: 'doughnut',
                data: {
                    labels: ['Correct (65.84%)', 'Non-Correct / Risk (34.16%)'],
                    datasets: [{
                        data: [181353, 94105],
                        backgroundColor: ['#10b981', '#f59e0b'],
                        borderWidth: 0,
                        hoverOffset: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => ` ${ctx.label}: ${ctx.raw.toLocaleString()} interactions`
                            }
                        }
                    },
                    cutout: '68%'
                }
            });
        }

        const ctxTypes = document.getElementById('chart-answer-types');
        if (ctxTypes && this.resultsData.dataset.answer_types) {
            const types = this.resultsData.dataset.answer_types;
            this.charts.answerTypes = new Chart(ctxTypes, {
                type: 'bar',
                data: {
                    labels: ['Algebra', 'Fill-in-1', 'Choose-1', 'Choose-n', 'Open Resp'],
                    datasets: [{
                        label: 'Interactions',
                        data: [types.algebra, types.fill_in_1, types.choose_1, types.choose_n, types.open_response],
                        backgroundColor: '#3b82f6',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            type: 'logarithmic',
                            grid: { color: c.grid },
                            title: { display: true, text: 'Count (log scale)' }
                        },
                        x: { grid: { display: false } }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => ` ${ctx.raw.toLocaleString()} questions`
                            }
                        }
                    }
                }
            });
        }
    },

    // 2. Single-Group Experiments
    renderSingleGroupChart() {
        const c = this.getThemeColors();
        const ctx = document.getElementById('chart-single-groups');
        if (!ctx || !this.resultsData.single_groups) return;

        const groups = this.resultsData.single_groups;
        this.charts.singleGroups = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: groups.map(g => `${g.group.split(':')[0]} (${g.selected})`),
                datasets: [
                    {
                        label: 'ROC-AUC (Higher is better)',
                        data: groups.map(g => g.roc_auc),
                        backgroundColor: '#3b82f6',
                        borderRadius: 6,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Log Loss (Lower is better)',
                        data: groups.map(g => g.log_loss),
                        backgroundColor: '#f59e0b',
                        borderRadius: 6,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        position: 'left',
                        min: 0.60,
                        max: 0.72,
                        grid: { color: c.grid },
                        title: { display: true, text: 'ROC-AUC' }
                    },
                    y1: {
                        position: 'right',
                        min: 0.55,
                        max: 0.65,
                        grid: { display: false },
                        title: { display: true, text: 'Log Loss' }
                    },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { position: 'top', labels: { boxWidth: 12 } }
                }
            }
        });
    },

    // 3. Ablation Study
    renderAblationChart() {
        const c = this.getThemeColors();
        const ctx = document.getElementById('chart-ablation');
        if (!ctx || !this.resultsData.ablation_study) return;

        const ablation = this.resultsData.ablation_study;
        this.charts.ablation = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ablation.map(a => a.config),
                datasets: [{
                    label: 'ROC-AUC Drop vs Full Model',
                    data: ablation.map(a => a.delta_auc),
                    backgroundColor: (ctx) => {
                        const val = ctx.raw;
                        if (val === 0) return '#10b981';
                        if (val < -0.015) return '#ef4444';
                        return '#f59e0b';
                    },
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: c.grid },
                        title: { display: true, text: 'Δ ROC-AUC (Relative to Full 0.7396)' }
                    },
                    y: { grid: { display: false } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` Δ ROC-AUC: ${ctx.raw.toFixed(4)}`
                        }
                    }
                }
            }
        });
    },

    // 4. Model Family Comparison
    renderModelComparisonChart() {
        const c = this.getThemeColors();
        const ctx = document.getElementById('chart-model-comparison');
        if (!ctx || !this.resultsData.model_comparison) return;

        const models = this.resultsData.model_comparison;
        this.charts.models = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models.map(m => m.model),
                datasets: [
                    {
                        label: 'ROC-AUC',
                        data: models.map(m => m.roc_auc),
                        backgroundColor: '#3b82f6',
                        borderRadius: 6
                    },
                    {
                        label: 'PR-AUC (Risk)',
                        data: models.map(m => m.risk_ap),
                        backgroundColor: '#10b981',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0.58,
                        max: 0.78,
                        grid: { color: c.grid },
                        title: { display: true, text: 'Score' }
                    },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { position: 'top', labels: { boxWidth: 12 } }
                }
            }
        });
    },

    // 5. Test Curves (ROC, PR, Calibration)
    renderTestCurves() {
        const c = this.getThemeColors();
        const curves = this.resultsData.curves;
        if (!curves) return;

        // ROC Curve
        const ctxRoc = document.getElementById('chart-roc-curve');
        if (ctxRoc && curves.roc) {
            this.charts.roc = new Chart(ctxRoc, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'XGBoost Test (AUC = 0.7468)',
                            data: curves.roc.map(p => ({ x: p.fpr, y: p.tpr })),
                            borderColor: '#3b82f6',
                            borderWidth: 2.5,
                            fill: false,
                            tension: 0.1,
                            pointRadius: 0
                        },
                        {
                            label: 'Random Chance (AUC = 0.50)',
                            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                            borderColor: c.muted,
                            borderWidth: 1.5,
                            borderDash: [5, 5],
                            fill: false,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { type: 'linear', min: 0, max: 1, grid: { color: c.grid }, title: { display: true, text: 'False Positive Rate' } },
                        y: { type: 'linear', min: 0, max: 1, grid: { color: c.grid }, title: { display: true, text: 'True Positive Rate' } }
                    },
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 12 } }
                    }
                }
            });
        }

        // PR Curve
        const ctxPr = document.getElementById('chart-pr-curve');
        if (ctxPr && curves.pr) {
            this.charts.pr = new Chart(ctxPr, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'XGBoost Risk (AP = 0.6241)',
                            data: curves.pr.map(p => ({ x: p.recall, y: p.precision })),
                            borderColor: '#10b981',
                            borderWidth: 2.5,
                            fill: false,
                            tension: 0.1,
                            pointRadius: 0
                        },
                        {
                            label: 'Prevalence Baseline (34.16%)',
                            data: [{ x: 0, y: 0.3416 }, { x: 1, y: 0.3416 }],
                            borderColor: c.muted,
                            borderWidth: 1.5,
                            borderDash: [5, 5],
                            fill: false,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { type: 'linear', min: 0, max: 1, grid: { color: c.grid }, title: { display: true, text: 'Recall (Failure Detection)' } },
                        y: { type: 'linear', min: 0, max: 1, grid: { color: c.grid }, title: { display: true, text: 'Precision' } }
                    },
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 12 } }
                    }
                }
            });
        }

        // Calibration Curve
        const ctxCal = document.getElementById('chart-calibration-curve');
        if (ctxCal && curves.calibration) {
            this.charts.cal = new Chart(ctxCal, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'XGBoost Binned Output',
                            data: curves.calibration.map(p => ({ x: p.pred, y: p.true })),
                            borderColor: '#8b5cf6',
                            backgroundColor: '#8b5cf6',
                            borderWidth: 2,
                            pointRadius: 4,
                            fill: false
                        },
                        {
                            label: 'Perfect Calibration',
                            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                            borderColor: c.muted,
                            borderWidth: 1.5,
                            borderDash: [5, 5],
                            fill: false,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { type: 'linear', min: 0, max: 1, grid: { color: c.grid }, title: { display: true, text: 'Mean Predicted Probability' } },
                        y: { type: 'linear', min: 0, max: 1, grid: { color: c.grid }, title: { display: true, text: 'Observed Fraction Correct' } }
                    },
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 12 } }
                    }
                }
            });
        }
    },

    // 6. Feature Importance Chart
    renderFeatureImportance() {
        const c = this.getThemeColors();
        const ctx = document.getElementById('chart-feature-importance');
        if (!ctx || !this.resultsData.feature_importance) return;

        const fi = this.resultsData.feature_importance;
        this.charts.featureImportance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: fi.map(f => f.name),
                datasets: [{
                    label: 'Relative Gain Importance (%)',
                    data: fi.map(f => f.importance),
                    backgroundColor: (ctx) => {
                        const item = fi[ctx.dataIndex];
                        if (!item) return '#3b82f6';
                        if (item.group === 'G2') return '#3b82f6';
                        if (item.group === 'G1') return '#10b981';
                        if (item.group === 'G4') return '#f59e0b';
                        if (item.group === 'G3') return '#8b5cf6';
                        return '#64748b';
                    },
                    borderRadius: 5
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: c.grid },
                        title: { display: true, text: 'Relative Gain (%)' }
                    },
                    y: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` ${ctx.raw.toFixed(2)}% relative gain contribution`
                        }
                    }
                }
            }
        });
    }
};

window.ChartManager = ChartManager;
document.addEventListener('DOMContentLoaded', () => ChartManager.init());
