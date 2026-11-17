const fs = require('fs');
const path = require('path');

console.log('=== RUNNING JS MODEL INFERENCE ON COMPREHENSIVE PARITY SUITE ===\n');

const modelJson = JSON.parse(fs.readFileSync(path.resolve('website/data/xgboost_trees.json'), 'utf8'));
const groundTruth = JSON.parse(fs.readFileSync(path.join(__dirname, 'parity_ground_truth.json'), 'utf8'));

// Exact replication of model_inference.js logic
const indexedTrees = modelJson.trees.map(tree => {
    const nodeMap = {};
    function indexNode(node) {
        if (node.split_condition !== undefined) node.f32_split = Math.fround(node.split_condition);
        if (node.leaf !== undefined) node.f32_leaf = Math.fround(node.leaf);
        nodeMap[node.nodeid] = node;
        if (node.children) node.children.forEach(indexNode);
    }
    indexNode(tree);
    return nodeMap;
});

function evaluateTree(nodeMap, features) {
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

function preprocess(sample) {
    const features = new Float32Array(11);
    for (let i = 0; i < 6; i++) {
        const featName = modelJson.numeric_features[i];
        const rawVal = Number(sample[featName] !== undefined ? sample[featName] : 0.0);
        const z = (rawVal - modelJson.scaler_means[i]) / modelJson.scaler_scales[i];
        features[i] = Math.fround(z);
    }
    const at = String(sample.answer_type || 'algebra');
    for (let j = 0; j < modelJson.categories.length; j++) {
        features[6 + j] = (modelJson.categories[j] === at) ? 1.0 : 0.0;
    }
    return features;
}

function predict(sample, threshold = 0.2880) {
    const features = preprocess(sample);
    let score = Math.fround(modelJson.base_margin);
    for (let t = 0; t < indexedTrees.length; t++) {
        score += evaluateTree(indexedTrees[t], features);
    }
    const pCorrect = 1.0 / (1.0 + Math.exp(-score));
    const pRisk = 1.0 - pCorrect;
    return {
        p_correct: pCorrect,
        p_risk: pRisk,
        flagged: pRisk >= threshold,
        logit_score: score
    };
}

let maxAbsDiff = 0.0;
let sumAbsDiff = 0.0;
let allConcordant = true;

console.log('| ID | Category | Python P(corr) | JS P(corr) | Abs Error | Flagged Py | Flagged JS | Status |');
console.log('|---|---|---|---|---|---|---|---|');

groundTruth.forEach(tc => {
    const jsRes = predict(tc.sample, 0.2880);
    const pyRes = tc.python_prediction;

    const diff = Math.abs(jsRes.p_correct - pyRes.p_correct);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
    sumAbsDiff += diff;

    const concordant = (jsRes.flagged === pyRes.flagged);
    if (!concordant) allConcordant = false;

    const status = (diff < 1e-5 && concordant) ? 'PASS' : 'FAIL';

    console.log(`| ${tc.id} | ${tc.category} | ${(pyRes.p_correct*100).toFixed(4)}% | ${(jsRes.p_correct*100).toFixed(4)}% | ${diff.toExponential(3)} | ${pyRes.flagged} | ${jsRes.flagged} | ${status} |`);
});

const meanAbsDiff = sumAbsDiff / groundTruth.length;

console.log('\n--- PARITY SUMMARY ---');
console.log(`Total Test Vectors: ${groundTruth.length}`);
console.log(`Max Absolute Error: ${maxAbsDiff.toExponential(4)}`);
console.log(`Mean Absolute Error: ${meanAbsDiff.toExponential(4)}`);
console.log(`Classification Concordance (tau=0.2880): ${allConcordant ? '100% (ALL CONCORDANT)' : 'MISMATCH DETECTED'}`);
console.log(`Float32 Threshold (< 1e-5): ${maxAbsDiff < 1e-5 ? 'PASS' : 'FAIL'}`);
