/**
 * Next Move AI — In-Browser XGBoost Inference Engine
 * Executes exact 500 decision trees in single-precision (Float32Array / Math.fround)
 * matching Python pipeline outputs to < 1e-6 error.
 */

class XGBoostInferenceEngine {
    constructor() {
        this.modelData = null;
        this.indexedTrees = null;
        this.isLoaded = false;
        this.loadPromise = null;
    }

    async loadModel(dataPath = 'data/xgboost_trees.json') {
        if (this.loadPromise) return this.loadPromise;

        this.loadPromise = (async () => {
            try {
                const response = await fetch(dataPath);
                if (!response.ok) {
                    throw new Error(`Failed to load model file (${response.status} ${response.statusText})`);
                }
                this.modelData = await response.json();
                this._indexTrees();
                this.isLoaded = true;
                return true;
            } catch (err) {
                console.error('Error loading XGBoost model:', err);
                throw err;
            }
        })();

        return this.loadPromise;
    }

    _indexTrees() {
        this.indexedTrees = this.modelData.trees.map(tree => {
            const nodeMap = {};
            function indexNode(node) {
                if (node.split_condition !== undefined) {
                    node.f32_split = Math.fround(node.split_condition);
                }
                if (node.leaf !== undefined) {
                    node.f32_leaf = Math.fround(node.leaf);
                }
                nodeMap[node.nodeid] = node;
                if (node.children) {
                    node.children.forEach(indexNode);
                }
            }
            indexNode(tree);
            return nodeMap;
        });
    }

    preprocess(sample) {
        if (!this.isLoaded) throw new Error('Model is not loaded yet');

        // 11 features: 6 standardized numeric + 5 one-hot categorical
        const features = new Float32Array(11);
        const zScores = [];

        // 6 numeric features
        for (let i = 0; i < 6; i++) {
            const featName = this.modelData.numeric_features[i];
            const rawVal = Number(sample[featName] !== undefined ? sample[featName] : 0.0);
            const z = (rawVal - this.modelData.scaler_means[i]) / this.modelData.scaler_scales[i];
            features[i] = Math.fround(z);
            zScores.push({
                feature: featName,
                raw: rawVal,
                mean: this.modelData.scaler_means[i],
                scale: this.modelData.scaler_scales[i],
                zScore: z
            });
        }

        // 1 categorical feature (answer_type)
        const at = String(sample.answer_type || 'algebra');
        const oneHot = [];
        for (let j = 0; j < this.modelData.categories.length; j++) {
            const cat = this.modelData.categories[j];
            const isMatch = (cat === at);
            features[6 + j] = isMatch ? 1.0 : 0.0;
            oneHot.push({ category: cat, active: isMatch });
        }

        return { features, zScores, oneHot };
    }

    _evaluateTree(nodeMap, features) {
        let curr = nodeMap[0];
        while (curr.leaf === undefined) {
            const featIdx = parseInt(curr.split.substring(1), 10);
            const val = features[featIdx];
            if (val < curr.f32_split) {
                curr = nodeMap[curr.yes];
            } else {
                curr = nodeMap[curr.no];
            }
        }
        return curr.f32_leaf;
    }

    predict(sample, threshold = 0.2880) {
        if (!this.isLoaded) throw new Error('Model is not loaded yet');

        const { features, zScores, oneHot } = this.preprocess(sample);

        let score = Math.fround(this.modelData.base_margin);
        for (let t = 0; t < this.indexedTrees.length; t++) {
            score += this._evaluateTree(this.indexedTrees[t], features);
        }

        const pCorrect = 1.0 / (1.0 + Math.exp(-score));
        const pRisk = 1.0 - pCorrect;
        const flagged = pRisk >= threshold;

        return {
            p_correct: pCorrect,
            p_risk: pRisk,
            threshold: threshold,
            flagged: flagged,
            logit_score: score,
            z_scores: zScores,
            one_hot: oneHot
        };
    }
}

window.XGBoostInference = new XGBoostInferenceEngine();
