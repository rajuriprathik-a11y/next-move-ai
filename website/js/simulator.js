/**
 * Next Move AI — Student Outcome Simulator & Model Explorer Controller
 * Connects UI controls to the verified in-browser Float32 XGBoost inference engine.
 */

const Simulator = {
    representativeQuestions: [],
    curatedScenarios: [],
    trainingPriors: null,
    questionMode: 'catalog', // 'catalog' | 'hypothetical'
    trainingPrior: 0.658624,
    smoothingAlpha: 10,

    async init() {
        try {
            // Load questions, scenarios, and fixed training priors
            const [qResp, sResp, pResp] = await Promise.all([
                fetch('data/representative_questions.json'),
                fetch('data/curated_scenarios.json'),
                fetch('data/training_priors.json')
            ]);
            this.representativeQuestions = await qResp.json();
            this.curatedScenarios = await sResp.json();
            this.trainingPriors = await pResp.json();

            // Ensure XGBoost model is loaded
            await window.XGBoostInference.loadModel();

            this.populateQuestionsDropdown();
            this.populateCuratedScenarios();
            this.bindEvents();

            // Run initial prediction
            this.runSimulation();
            this.runModelExplorer();
        } catch (err) {
            console.error('Failed to initialize Simulator:', err);
        }
    },

    getSkillPriors(skillIdStr) {
        if (!skillIdStr || skillIdStr === 'null' || skillIdStr === 'undefined' || skillIdStr === 'unseen' || skillIdStr === 'none') {
            // Cold start / missing skill fallback from original implementation (src/feature_engineering.py)
            return {
                g2Prior: 0.50,
                g3Prior: 0.50,
                isFallback: true,
                label: 'Unseen / Missing Skill (Fallback P = 0.50)'
            };
        }

        const skills = String(skillIdStr).split('_').map(s => s.trim()).filter(Boolean);
        if (skills.length === 0) {
            return {
                g2Prior: 0.50,
                g3Prior: 0.50,
                isFallback: true,
                label: 'Missing Skill (Fallback P = 0.50)'
            };
        }

        const skillsData = this.trainingPriors?.g2_same_skill_history?.skills || {};
        const poolN = this.trainingPriors?.g3_other_skill_history?.annotated_pool_count || 206534;
        const poolC = this.trainingPriors?.g3_other_skill_history?.annotated_pool_correct || 135861;

        let totalWeight = 0;
        let totalCorrect = 0;

        for (const s of skills) {
            if (skillsData[s]) {
                totalWeight += skillsData[s].fractional_weight;
                totalCorrect += skillsData[s].fractional_correct;
            }
        }

        if (totalWeight <= 0) {
            // Unseen skill not present in training pool
            return {
                g2Prior: 0.50,
                g3Prior: 0.50,
                isFallback: true,
                label: `Unseen Skill [${skills.join(', ')}] (Fallback P = 0.50)`
            };
        }

        const g2Prior = totalCorrect / totalWeight;
        const otherWeight = poolN - totalWeight;
        const otherCorrect = poolC - totalCorrect;
        const g3Prior = otherWeight > 0 ? (otherCorrect / otherWeight) : 0.50;

        return {
            g2Prior,
            g3Prior,
            isFallback: false,
            skillsCount: skills.length,
            weight: totalWeight,
            label: skills.length > 1 ? `Multi-Skill [${skills.join(', ')}]` : `Skill #${skills[0]}`
        };
    },

    populateQuestionsDropdown() {
        const select = document.getElementById('sim-question-select');
        if (!select) return;

        select.innerHTML = '';
        this.representativeQuestions.forEach((q, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = `Problem #${q.problem_id} — ${q.tier} (${(q.smoothed_accuracy * 100).toFixed(1)}% acc, ${q.answer_type}, skill ${q.skill_id})`;
            select.appendChild(opt);
        });

        // Set default to average difficulty question
        select.selectedIndex = Math.min(5, this.representativeQuestions.length - 1);
        this.updateSelectedQuestionDetails();
    },

    updateSelectedQuestionDetails() {
        const select = document.getElementById('sim-question-select');
        const detailsEl = document.getElementById('sim-selected-q-details');
        if (!select || !detailsEl) return;

        const q = this.representativeQuestions[select.selectedIndex];
        if (!q) return;

        const skillInfo = this.getSkillPriors(q.skill_id);
        const g2Pct = (skillInfo.g2Prior * 100).toFixed(1);
        const g3Pct = (skillInfo.g3Prior * 100).toFixed(1);

        detailsEl.innerHTML = `
            <div class="q-detail-pill"><strong>Problem:</strong> #${q.problem_id}</div>
            <div class="q-detail-pill"><strong>Difficulty Tier:</strong> <span class="badge tier-${q.tier.toLowerCase().replace(' ', '-')}">${q.tier}</span></div>
            <div class="q-detail-pill"><strong>Question Acc (G4):</strong> ${(q.smoothed_accuracy * 100).toFixed(1)}% (${q.interaction_count} attempts)</div>
            <div class="q-detail-pill"><strong>Target Skill Prior (G2):</strong> ${g2Pct}% (${skillInfo.label})</div>
            <div class="q-detail-pill"><strong>Other-Skill Prior (G3):</strong> ${g3Pct}%</div>
            <div class="q-detail-pill"><strong>Format:</strong> <code>${q.answer_type}</code></div>
        `;

        // Automatically match format dropdown to question format if desired
        const formatSelect = document.getElementById('sim-format-select');
        if (formatSelect) {
            formatSelect.value = q.answer_type;
        }
    },

    populateCuratedScenarios() {
        const container = document.getElementById('curated-scenarios-container');
        if (!container) return;

        container.innerHTML = '';
        this.curatedScenarios.forEach((sc, idx) => {
            const card = document.createElement('button');
            card.type = 'button';
            const isRisk = !!sc.prediction.flagged;
            const pRiskPct = (sc.prediction.p_risk * 100).toFixed(1);
            const isSelected = (this.activeScenarioId === sc.id);

            // Clean title and extract parenthetical tag if present
            let cleanTitle = sc.title.replace(/^\d+\.\s*/, '');
            let tag = '';
            if (cleanTitle.includes('(')) {
                const match = cleanTitle.match(/^(.*?)\s*\((.*?)\)$/);
                if (match) {
                    cleanTitle = match[1].trim();
                    tag = match[2].trim();
                }
            }

            const u = sc.unlogged_counts || { total_questions: 0, total_correct: 0, same_skill_questions: 0, same_skill_correct: 0 };
            const overallAcc = u.total_questions > 0 ? Math.round((u.total_correct / u.total_questions) * 100) : null;
            const skillAcc = u.same_skill_questions > 0 ? Math.round((u.same_skill_correct / u.same_skill_questions) * 100) : null;

            card.className = `scenario-card ${isRisk ? 'status-risk' : 'status-safe'} ${isSelected ? 'is-selected' : ''}`;
            card.setAttribute('data-scenario-id', sc.id);
            card.setAttribute('aria-pressed', isSelected ? 'true' : 'false');
            card.setAttribute('title', `Simulate archetype: ${cleanTitle}`);

            card.innerHTML = `
                <div class="scenario-meta">
                    <span class="archetype-badge">Archetype 0${idx + 1}</span>
                    <span class="scenario-status-pill ${isRisk ? 'pill-risk' : 'pill-safe'}">
                        <span class="status-dot"></span>
                        <span class="status-label">${isRisk ? 'Flagged Risk' : 'Cleared'}</span>
                        <span class="status-val">${pRiskPct}%</span>
                    </span>
                </div>
                <div class="scenario-heading-group">
                    <h4 class="scenario-title">${cleanTitle}</h4>
                    ${tag ? `<span class="scenario-tag">${tag}</span>` : ''}
                </div>
                <p class="scenario-desc">${sc.description}</p>
                <div class="scenario-stats-row">
                    <div class="stat-item" title="Overall Practice History">
                        <span class="stat-label">Overall:</span>
                        <span class="stat-val">${u.total_questions > 0 ? `${u.total_correct}/${u.total_questions} (${overallAcc}%)` : '0 Qs (Prior)'}</span>
                    </div>
                    <div class="stat-item" title="Target Skill Practice History">
                        <span class="stat-label">Target Skill:</span>
                        <span class="stat-val">${u.same_skill_questions > 0 ? `${u.same_skill_correct}/${u.same_skill_questions} (${skillAcc}%)` : '0 Qs (Prior)'}</span>
                    </div>
                </div>
                <div class="scenario-footer">
                    <span class="cue-default">
                        <span>Try this scenario</span>
                        <svg class="cue-arrow" width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M6 12L10 8L6 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
                    </span>
                    <span class="cue-active">
                        <span class="active-dot"></span>
                        <span>Currently Active</span>
                    </span>
                </div>
            `;

            card.addEventListener('click', () => {
                this.loadScenario(sc);
            });
            container.appendChild(card);
        });
    },

    updateSelectedArchetypeUI(selectedId) {
        const container = document.getElementById('curated-scenarios-container');
        if (!container) return;
        container.querySelectorAll('.scenario-card').forEach(card => {
            const cardId = card.getAttribute('data-scenario-id');
            const isMatch = (cardId === selectedId);
            card.classList.toggle('is-selected', isMatch);
            card.setAttribute('aria-pressed', isMatch ? 'true' : 'false');
        });
    },

    clearActiveArchetype() {
        this.activeScenarioId = null;
        const container = document.getElementById('curated-scenarios-container');
        if (!container) return;
        container.querySelectorAll('.scenario-card').forEach(card => {
            card.classList.remove('is-selected');
            card.setAttribute('aria-pressed', 'false');
        });
    },

    loadScenario(scenario) {
        this.activeScenarioId = scenario.id;
        this.updateSelectedArchetypeUI(scenario.id);

        // Switch to catalog mode if needed or set inputs
        const u = scenario.unlogged_counts;
        if (u) {
            this.setVal('sim-total-n', u.total_questions);
            this.setVal('sim-total-c', u.total_correct);
            this.setVal('sim-skill-n', u.same_skill_questions);
            this.setVal('sim-skill-c', u.same_skill_correct);

            // Sync sliders
            this.syncSliderWithNumber('sim-total-n', 'sim-total-n-range');
            this.syncSliderWithNumber('sim-total-c', 'sim-total-c-range');
            this.syncSliderWithNumber('sim-skill-n', 'sim-skill-n-range');
            this.syncSliderWithNumber('sim-skill-c', 'sim-skill-c-range');
        }

        const formatSelect = document.getElementById('sim-format-select');
        if (formatSelect && scenario.inputs.answer_type) {
            formatSelect.value = scenario.inputs.answer_type;
        }

        // Smoothly ensure simulator results are visible without abrupt jumping
        const simEl = document.getElementById('student-simulator');
        if (simEl) {
            simEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        this.runSimulation();
    },

    bindEvents() {
        // Sync number inputs and range sliders
        const pairs = [
            ['sim-total-n', 'sim-total-n-range'],
            ['sim-total-c', 'sim-total-c-range'],
            ['sim-skill-n', 'sim-skill-n-range'],
            ['sim-skill-c', 'sim-skill-c-range'],
            ['hyp-acc-slider', 'hyp-acc-val'],
            ['hyp-exp-slider', 'hyp-exp-val']
        ];

        pairs.forEach(([numId, rangeId]) => {
            const num = document.getElementById(numId);
            const range = document.getElementById(rangeId);
            if (num && range) {
                num.addEventListener('input', () => {
                    range.value = num.value;
                    this.runSimulation();
                });
                range.addEventListener('input', () => {
                    num.value = range.value;
                    this.runSimulation();
                });
            }
        });

        // Question mode radio toggle
        document.querySelectorAll('input[name="sim-q-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.questionMode = e.target.value;
                const catGroup = document.getElementById('sim-q-catalog-group');
                const hypGroup = document.getElementById('sim-q-hypothetical-group');
                if (catGroup && hypGroup) {
                    if (this.questionMode === 'catalog') {
                        catGroup.style.display = 'block';
                        hypGroup.style.display = 'none';
                    } else {
                        catGroup.style.display = 'none';
                        hypGroup.style.display = 'block';
                    }
                }
                this.runSimulation();
            });
        });

        // Question catalog select change
        const qSelect = document.getElementById('sim-question-select');
        if (qSelect) {
            qSelect.addEventListener('change', () => {
                this.updateSelectedQuestionDetails();
                this.runSimulation();
            });
        }

        // Format select change
        const fSelect = document.getElementById('sim-format-select');
        if (fSelect) {
            fSelect.addEventListener('change', () => this.runSimulation());
        }

        // Hypothetical inputs change
        ['hyp-acc-slider', 'hyp-exp-slider', 'hyp-skill-prior-slider'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('input', () => this.runSimulation());
            }
        });

        // Hypothetical skill preset change
        const hypPresetSelect = document.getElementById('hyp-skill-preset-select');
        const hypPriorSlider = document.getElementById('hyp-skill-prior-slider');
        if (hypPresetSelect) {
            hypPresetSelect.addEventListener('change', () => {
                const val = hypPresetSelect.value;
                if (val === 'skill_27' && hypPriorSlider) hypPriorSlider.value = 63;
                else if (val === 'skill_76' && hypPriorSlider) hypPriorSlider.value = 38;
                else if (val === 'skill_18' && hypPriorSlider) hypPriorSlider.value = 76;
                else if (val === 'skill_16_17' && hypPriorSlider) hypPriorSlider.value = 53;
                else if (val === 'unseen' && hypPriorSlider) hypPriorSlider.value = 50;
                this.runSimulation();
            });
        }

        if (hypPriorSlider) {
            hypPriorSlider.addEventListener('input', () => {
                if (hypPresetSelect && hypPresetSelect.value !== 'custom') {
                    const val = parseInt(hypPriorSlider.value, 10);
                    if (![63, 38, 76, 53, 50].includes(val)) {
                        hypPresetSelect.value = 'custom';
                    }
                }
                this.runSimulation();
            });
        }

        // Model Explorer Sliders
        const expFeatures = [
            'exp-f0', 'exp-f1', 'exp-f2', 'exp-f3', 'exp-f4', 'exp-f5', 'exp-f6'
        ];
        expFeatures.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('input', () => this.runModelExplorer());
            }
        });

        // Reset buttons
        const simReset = document.getElementById('sim-reset-btn');
        if (simReset) {
            simReset.addEventListener('click', () => {
                this.clearActiveArchetype();
                this.setVal('sim-total-n', 30);
                this.setVal('sim-total-c', 21);
                this.setVal('sim-skill-n', 8);
                this.setVal('sim-skill-c', 5);
                this.syncAllSimSliders();
                this.runSimulation();
            });
        }

        const expReset = document.getElementById('exp-reset-btn');
        if (expReset) {
            expReset.addEventListener('click', () => {
                this.setVal('exp-f0', 0.674);
                this.setVal('exp-f1', 0.629);
                this.setVal('exp-f2', 1.632);
                this.setVal('exp-f3', 0.674);
                this.setVal('exp-f4', 0.663);
                this.setVal('exp-f5', 2.186);
                const f6 = document.getElementById('exp-f6');
                if (f6) f6.value = 'algebra';
                this.runModelExplorer();
            });
        }
    },

    syncSliderWithNumber(numId, rangeId) {
        const num = document.getElementById(numId);
        const range = document.getElementById(rangeId);
        if (num && range) range.value = num.value;
    },

    syncAllSimSliders() {
        this.syncSliderWithNumber('sim-total-n', 'sim-total-n-range');
        this.syncSliderWithNumber('sim-total-c', 'sim-total-c-range');
        this.syncSliderWithNumber('sim-skill-n', 'sim-skill-n-range');
        this.syncSliderWithNumber('sim-skill-c', 'sim-skill-c-range');
    },

    setVal(id, val) {
        const el = document.getElementById(id);
        if (el) el.value = val;
    },

    runSimulation() {
        // Read raw user inputs
        let totalN = parseInt(document.getElementById('sim-total-n')?.value, 10) || 0;
        let totalC = parseInt(document.getElementById('sim-total-c')?.value, 10) || 0;
        let skillN = parseInt(document.getElementById('sim-skill-n')?.value, 10) || 0;
        let skillC = parseInt(document.getElementById('sim-skill-c')?.value, 10) || 0;
        const format = document.getElementById('sim-format-select')?.value || 'algebra';

        // Invariant Validation
        const warningEl = document.getElementById('sim-validation-warning');
        const warnings = [];

        if (totalC > totalN) {
            warnings.push(`Total correct (${totalC}) cannot exceed total questions (${totalN}). Clamping total correct.`);
            totalC = totalN;
            this.setVal('sim-total-c', totalC);
            this.syncSliderWithNumber('sim-total-c', 'sim-total-c-range');
        }

        if (skillN > totalN) {
            warnings.push(`Same-skill questions (${skillN}) cannot exceed total questions (${totalN}). Clamping same-skill count.`);
            skillN = totalN;
            this.setVal('sim-skill-n', skillN);
            this.syncSliderWithNumber('sim-skill-n', 'sim-skill-n-range');
        }

        if (skillC > skillN) {
            warnings.push(`Same-skill correct (${skillC}) cannot exceed same-skill questions (${skillN}). Clamping same-skill correct.`);
            skillC = skillN;
            this.setVal('sim-skill-c', skillC);
            this.syncSliderWithNumber('sim-skill-c', 'sim-skill-c-range');
        }

        if (skillC > totalC) {
            warnings.push(`Same-skill correct (${skillC}) cannot exceed total correct (${totalC}). Clamping same-skill correct.`);
            skillC = totalC;
            this.setVal('sim-skill-c', skillC);
            this.syncSliderWithNumber('sim-skill-c', 'sim-skill-c-range');
        }

        if (warningEl) {
            if (warnings.length > 0) {
                warningEl.style.display = 'block';
                warningEl.innerHTML = `<strong>Input Invariant Adjusted:</strong> ${warnings.join('<br>')}`;
            } else {
                warningEl.style.display = 'none';
            }
        }

        // Calculate other-skill numbers via clean residual (disclosing single-skill assumption)
        const otherN = Math.max(0, totalN - skillN);
        const otherC = Math.max(0, totalC - skillC);

        // Determine exact Bayesian Smoothing Priors from Training Pool
        const alpha = this.smoothingAlpha;
        const g1Prior = this.trainingPriors?.g1_overall_history?.final_global_prior || 0.658624;
        let g2Prior = 0.628947;           // average training-pool skill prior
        let g3Prior = 0.658558;           // average training-pool other-skill prior
        let questionSmoothedAcc = 0.6626;
        let logQuestionInteractions = 2.7081;
        let questionLabel = '';

        if (this.questionMode === 'catalog') {
            const qSelect = document.getElementById('sim-question-select');
            const qIdx = qSelect ? qSelect.selectedIndex : 0;
            const q = this.representativeQuestions[qIdx];
            if (q) {
                const skillInfo = this.getSkillPriors(q.skill_id);
                g2Prior = skillInfo.g2Prior;
                g3Prior = skillInfo.g3Prior;
                questionSmoothedAcc = q.smoothed_accuracy;
                logQuestionInteractions = q.log_interactions;
                questionLabel = `Problem #${q.problem_id} (${q.tier}, ${skillInfo.label})`;
            }
        } else {
            // Hypothetical mode
            const hypPreset = document.getElementById('hyp-skill-preset-select')?.value || 'custom';
            let hypSkillId = 'custom';
            if (hypPreset === 'skill_27') hypSkillId = '27';
            else if (hypPreset === 'skill_76') hypSkillId = '76';
            else if (hypPreset === 'skill_18') hypSkillId = '18';
            else if (hypPreset === 'skill_16_17') hypSkillId = '16_17';
            else if (hypPreset === 'unseen') hypSkillId = 'unseen';

            if (hypSkillId !== 'custom') {
                const skillInfo = this.getSkillPriors(hypSkillId);
                g2Prior = skillInfo.g2Prior;
                g3Prior = skillInfo.g3Prior;
                questionLabel = `Hypothetical question with ${skillInfo.label}`;
            } else {
                const hypSkillPrior = parseFloat(document.getElementById('hyp-skill-prior-slider')?.value) || 63;
                g2Prior = hypSkillPrior / 100.0;
                // Empirical other-skill prior calculation matching src/feature_engineering.py
                const poolN = this.trainingPriors?.g3_other_skill_history?.annotated_pool_count || 206534;
                const poolC = this.trainingPriors?.g3_other_skill_history?.annotated_pool_correct || 135861;
                const estWeight = 3500;
                const estCorrect = estWeight * g2Prior;
                g3Prior = (poolC - estCorrect) / (poolN - estWeight);
                questionLabel = `Hypothetical question (custom skill prior ${hypSkillPrior}%)`;
            }

            const hypAcc = parseFloat(document.getElementById('hyp-acc-slider')?.value) || 65;
            const hypExp = parseInt(document.getElementById('hyp-exp-slider')?.value, 10) || 0;

            // Apply G4 Bayesian smoothing matching src/feature_engineering.py
            if (hypExp === 0) {
                // Unseen question cold-start
                questionSmoothedAcc = g1Prior;
                logQuestionInteractions = 0.0;
            } else {
                const rawCorrect = (hypAcc / 100.0) * hypExp;
                questionSmoothedAcc = (rawCorrect + alpha * g1Prior) / (hypExp + alpha);
                logQuestionInteractions = Math.log1p(hypExp);
            }

            // Update badge displays
            const hypAccDisplay = document.getElementById('hyp-acc-val');
            const hypExpDisplay = document.getElementById('hyp-exp-val');
            const hypSkillDisplay = document.getElementById('hyp-skill-prior-val');
            if (hypAccDisplay) hypAccDisplay.textContent = `${hypAcc}% (Smoothed: ${(questionSmoothedAcc * 100).toFixed(1)}%)`;
            if (hypExpDisplay) hypExpDisplay.textContent = hypExp === 0 ? '0 attempts (Cold Start)' : `${hypExp} attempts`;
            if (hypSkillDisplay) hypSkillDisplay.textContent = `${(g2Prior * 100).toFixed(1)}% (P_G2)`;
        }

        // Apply Bayesian shrinkage (m-estimate)
        const overallSmoothedAcc = (totalC + alpha * g1Prior) / (totalN + alpha);
        const sameSkillSmoothedAcc = (skillC + alpha * g2Prior) / (skillN + alpha);
        const logSameSkillInteractions = Math.log1p(skillN);
        const otherSkillSmoothedAcc = (otherC + alpha * g3Prior) / (otherN + alpha);

        // Construct exact 7-feature payload
        const samplePayload = {
            overall_smoothed_accuracy: overallSmoothedAcc,
            weighted_same_skill_smoothed_accuracy: sameSkillSmoothedAcc,
            log_weighted_same_skill_previous_interactions: logSameSkillInteractions,
            weighted_other_skill_smoothed_accuracy: otherSkillSmoothedAcc,
            question_smoothed_accuracy: questionSmoothedAcc,
            log_question_previous_interactions: logQuestionInteractions,
            answer_type: format
        };

        // Run in-browser prediction
        const result = window.XGBoostInference.predict(samplePayload, 0.2880);

        // Update UI Outputs
        this.renderSimulationResult(result, samplePayload, {
            totalN, totalC, skillN, skillC, otherN, otherC
        }, {
            g1: g1Prior, g2: g2Prior, g3: g3Prior, questionLabel
        });
    },

    renderSimulationResult(result, features, counts, priors = {}) {
        const pCorrPct = (result.p_correct * 100).toFixed(1);
        const pRiskPct = (result.p_risk * 100).toFixed(1);

        // Update probability meters
        const pCorrEl = document.getElementById('sim-p-correct');
        const pRiskEl = document.getElementById('sim-p-risk');
        const decisionBadge = document.getElementById('sim-decision-badge');
        const riskBar = document.getElementById('sim-risk-bar-fill');

        if (pCorrEl) pCorrEl.textContent = `${pCorrPct}%`;
        if (pRiskEl) pRiskEl.textContent = `${pRiskPct}%`;

        if (riskBar) {
            riskBar.style.width = `${pRiskPct}%`;
            riskBar.className = 'progress-bar-fill ' + (result.flagged ? 'bg-danger' : 'bg-success');
        }

        if (decisionBadge) {
            if (result.flagged) {
                decisionBadge.className = 'result-badge badge-flagged';
                decisionBadge.innerHTML = `<span class="badge-icon">⚠️</span> FLAG FOR EDUCATIONAL ASSISTANCE (Risk ${pRiskPct}% ≥ 28.8%)`;
            } else {
                decisionBadge.className = 'result-badge badge-cleared';
                decisionBadge.innerHTML = `<span class="badge-icon">✓</span> LOW RISK / CLEARED (Risk ${pRiskPct}% < 28.8%)`;
            }
        }

        // Populate 7-Feature Vector Table
        const tbody = document.getElementById('sim-feature-table-body');
        if (tbody) {
            const g1P = priors.g1 !== undefined ? priors.g1.toFixed(4) : '0.6586';
            const g2P = priors.g2 !== undefined ? priors.g2.toFixed(4) : '0.6289';
            const g3P = priors.g3 !== undefined ? priors.g3.toFixed(4) : '0.6586';

            tbody.innerHTML = `
                <tr>
                    <td><code>overall_smoothed_accuracy</code></td>
                    <td>G1</td>
                    <td><strong>${features.overall_smoothed_accuracy.toFixed(4)}</strong></td>
                    <td>From ${counts.totalC}/${counts.totalN} correct + \\(\\alpha=10\\) smoothing (Global prior \\(P = ${g1P}\\))</td>
                </tr>
                <tr>
                    <td><code>weighted_same_skill_smoothed_accuracy</code></td>
                    <td>G2</td>
                    <td><strong>${features.weighted_same_skill_smoothed_accuracy.toFixed(4)}</strong></td>
                    <td>From ${counts.skillC}/${counts.skillN} correct on topic (Skill prior \\(P = ${g2P}\\))</td>
                </tr>
                <tr>
                    <td><code>log_weighted_same_skill_previous_interactions</code></td>
                    <td>G2</td>
                    <td><strong>${features.log_weighted_same_skill_previous_interactions.toFixed(4)}</strong></td>
                    <td>\\(\\ln(1 + ${counts.skillN})\\)</td>
                </tr>
                <tr>
                    <td><code>weighted_other_skill_smoothed_accuracy</code></td>
                    <td>G3</td>
                    <td><strong>${features.weighted_other_skill_smoothed_accuracy.toFixed(4)}</strong></td>
                    <td>From ${counts.otherC}/${counts.otherN} correct in other subjects (Other-skill prior \\(P = ${g3P}\\))</td>
                </tr>
                <tr>
                    <td><code>question_smoothed_accuracy</code></td>
                    <td>G4</td>
                    <td><strong>${features.question_smoothed_accuracy.toFixed(4)}</strong></td>
                    <td>${this.questionMode === 'catalog' ? (priors.questionLabel || 'Training pool problem statistic') : 'Hypothetical difficulty assumption'}</td>
                </tr>
                <tr>
                    <td><code>log_question_previous_interactions</code></td>
                    <td>G4</td>
                    <td><strong>${features.log_question_previous_interactions.toFixed(4)}</strong></td>
                    <td>\\(\\ln(1 + N_{\\text{exposure}})\\)</td>
                </tr>
                <tr>
                    <td><code>answer_type</code></td>
                    <td>G4</td>
                    <td><strong><code>${features.answer_type}</code></strong></td>
                    <td>One-hot encoded (5 indicator classes)</td>
                </tr>
            `;
            if (window.renderMath) window.renderMath(tbody);
        }
    },

    runModelExplorer() {
        const f0 = parseFloat(document.getElementById('exp-f0')?.value) || 0.674;
        const f1 = parseFloat(document.getElementById('exp-f1')?.value) || 0.629;
        const f2 = parseFloat(document.getElementById('exp-f2')?.value) || 1.632;
        const f3 = parseFloat(document.getElementById('exp-f3')?.value) || 0.674;
        const f4 = parseFloat(document.getElementById('exp-f4')?.value) || 0.663;
        const f5 = parseFloat(document.getElementById('exp-f5')?.value) || 2.186;
        const f6 = document.getElementById('exp-f6')?.value || 'algebra';

        // Update slider value badges
        this.setText('exp-f0-val', f0.toFixed(3));
        this.setText('exp-f1-val', f1.toFixed(3));
        this.setText('exp-f2-val', `${f2.toFixed(3)} (N ≈ ${Math.round(Math.expm1(f2))})`);
        this.setText('exp-f3-val', f3.toFixed(3));
        this.setText('exp-f4-val', f4.toFixed(3));
        this.setText('exp-f5-val', `${f5.toFixed(3)} (N ≈ ${Math.round(Math.expm1(f5))})`);

        const rawSample = {
            overall_smoothed_accuracy: f0,
            weighted_same_skill_smoothed_accuracy: f1,
            log_weighted_same_skill_previous_interactions: f2,
            weighted_other_skill_smoothed_accuracy: f3,
            question_smoothed_accuracy: f4,
            log_question_previous_interactions: f5,
            answer_type: f6
        };

        const result = window.XGBoostInference.predict(rawSample, 0.2880);

        // Update Explorer Displays
        this.setText('exp-res-corr', `${(result.p_correct * 100).toFixed(2)}%`);
        this.setText('exp-res-risk', `${(result.p_risk * 100).toFixed(2)}%`);
        this.setText('exp-res-logit', result.logit_score.toFixed(4));

        const badge = document.getElementById('exp-res-badge');
        if (badge) {
            if (result.flagged) {
                badge.className = 'badge badge-risk';
                badge.textContent = 'FLAGGED FOR ATTENTION';
            } else {
                badge.className = 'badge badge-safe';
                badge.textContent = 'CLEARED / LOW RISK';
            }
        }

        // Render Z-Scores & One-Hot Output
        const zBody = document.getElementById('exp-zscore-table-body');
        if (zBody && result.z_scores) {
            zBody.innerHTML = result.z_scores.map(z => `
                <tr>
                    <td><code>${z.feature}</code></td>
                    <td>${z.raw.toFixed(4)}</td>
                    <td>${z.mean.toFixed(4)}</td>
                    <td>${z.scale.toFixed(4)}</td>
                    <td><strong>${z.zScore.toFixed(4)}</strong></td>
                </tr>
            `).join('');
        }

        const oneHotContainer = document.getElementById('exp-onehot-display');
        if (oneHotContainer && result.one_hot) {
            oneHotContainer.innerHTML = result.one_hot.map(oh => `
                <span class="one-hot-pill ${oh.active ? 'active' : ''}">
                    ${oh.category}: <strong>${oh.active ? '1.0' : '0.0'}</strong>
                </span>
            `).join('');
        }
    },

    setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }
};

window.Simulator = Simulator;
document.addEventListener('DOMContentLoaded', () => Simulator.init());
