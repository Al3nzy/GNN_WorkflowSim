/**
 * Copyright 2012-2013 University Of Southern California
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.workflowsim.planning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.cloudbus.cloudsim.Log;
import org.workflowsim.CondorVM;
import org.workflowsim.Task;
import org.workflowsim.Job;
import org.workflowsim.Task;
import org.workflowsim.WorkflowEngine;
import org.workflowsim.planning.LIWSAGNNPlanningAlgorithm; // or org.workflowsim.examples.planning.LIWSAGNNPlanningAlgorithm depending on package placement
/**
 * LIWSA-GNN: LIWSA extended with a pure-Java graph neural network warm-start
 * predictor. Follow-up work to LIWSA-ML (Section "LIWSA-ML" of the base
 * paper), developed for a second, separate paper on cross-workflow
 * generalisation.
 *
 * DESIGN NOTE, important and different from LIWSA-ML's OLS predictor: the
 * GNN was trained to score a COMPLETE genotype (the full DAG structure plus
 * a full task-to-VM assignment) with a single (makespan, cost) prediction,
 * not to score individual (task, VM) pairs in isolation the way the OLS
 * model does. This is not an oversight; it is what lets the model use
 * direction-aware message passing over the actual DAG edges, which a
 * per-pair model structurally cannot do. Consequently this class uses the
 * GNN as a fast FITNESS ORACLE: it samples a batch of candidate random
 * genotypes, scores every one of them with a single GNN forward pass, and
 * seeds the population with the best-scoring candidates, rather than
 * building a genotype task-by-task the way LIWSA-ML's softmax sampling
 * does.
 *
 * VALIDATION, important for anyone maintaining this class: the forward
 * pass implemented below (dagMessagePassingLayer, linear, relu, mean
 * pooling, MLP head) was cross-checked against the real trained PyTorch
 * model on held-out samples before being ported here; a plain-Python
 * (no-torch) reference implementation matched PyTorch's output to within
 * 1.2e-7 absolute error. This Java port is a direct, mechanical
 * translation of that verified reference, not a re-derivation, and should
 * be re-validated the same way (compare against the Python reference on a
 * few known samples) if the architecture or weights ever change.
 *
 * TRAINING SCOPE, stated honestly: the bundled weights were trained on all
 * 20 Pegasus benchmark workflows across five families (Montage, CyberShake,
 * Sipht, Epigenomics, Inspiral). A held-out evaluation showed this model
 * generalises well to UNSEEN SCALES of a FAMILIAR family (e.g. trained on
 * small/medium instances, evaluated on 1000-task instances of the same
 * families: 5/5 random seeds showed the GNN beating a structure-blind
 * baseline by 2-19x on held-out prediction error). A separate evaluation
 * held out an ENTIRE family (e.g. trained on four families, tested on a
 * fifth never seen at any scale) and found the GNN performing WORSE than
 * the structure-blind baseline in all 5 cases. This predictor is therefore
 * only recommended for the five workflow families it was trained on; it is
 * not validated for use on a genuinely novel workflow topology, and should
 * not be presented or deployed as such.
 */
public class LIWSAGNNPlanningAlgorithm extends LIWSAPlanningAlgorithm {

    // ---- GNN architecture (must match the trained weights exactly) ----
    private static final int IN_DIM = 6;
    private static final int HIDDEN_DIM = 32;
    private static final int N_LAYERS = 3;

    // ---- candidate-sampling parameters ----
    public static int CONFIG_NUM_CANDIDATES = 80;
    public static int CONFIG_NUM_SEEDS_FROM_GNN = 4;
    public static String CONFIG_WEIGHTS_PATH = "gnn_weights.txt";

    private static Map<String, double[][]> weightMatrices;
    private static Map<String, double[]> biasVectors;
    private static boolean weightsLoaded = false;

    public LIWSAGNNPlanningAlgorithm() {
        super();
    }

    // -----------------------------------------------------------------
    // Weight loading (plain text, no JSON library; see gnn_weights.txt,
    // exported by export_weights.py / verify_plain_forward.py from the
    // trained PyTorch model. Format: one line per tensor,
    // "<name> <count> <v1> <v2> ... <vN>", flattened row-major).
    // -----------------------------------------------------------------
    private static synchronized void loadWeightsIfNeeded() {
        if (weightsLoaded) {
            return;
        }
        weightMatrices = new HashMap<>();
        biasVectors = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(CONFIG_WEIGHTS_PATH))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                String[] parts = line.split("\\s+");
                String name = parts[0];
                int count = Integer.parseInt(parts[1]);
                double[] flat = new double[count];
                for (int i = 0; i < count; i++) {
                    flat[i] = Double.parseDouble(parts[2 + i]);
                }

                if (name.endsWith(".bias")) {
                    biasVectors.put(name, flat);
                } else {
                    // weight matrices: reshape flat[out*in] -> [out][in].
                    // out_dim is inferred from the paired bias vector's
                    // length, which is always loaded together with its
                    // weight in this file (bias lines appear immediately
                    // after their weight line in the export).
                    weightMatrices.put(name, reshapeLater(flat));
                }
            }
        } catch (IOException e) {
            Log.printLine("LIWSA-GNN: FAILED to load weights from " + CONFIG_WEIGHTS_PATH
                + " (" + e.getMessage() + "). GNN seeding will be skipped; falling back to "
                + "random initialisation only.");
            weightMatrices = null;
            biasVectors = null;
            weightsLoaded = true;
            return;
        }

        // Second pass: now that every bias vector is known, reshape each
        // weight matrix's flat storage into [out][in] using the matching
        // bias vector's length as out_dim.
        Map<String, double[][]> reshaped = new HashMap<>();
        for (Map.Entry<String, double[][]> e : weightMatrices.entrySet()) {
            String weightName = e.getKey();
            String biasName = weightName.substring(0, weightName.length() - ".weight".length()) + ".bias";
            double[] bias = biasVectors.get(biasName);
            double[] flat = e.getValue()[0]; // stashed flat array, see reshapeLater()
            int outDim = bias.length;
            int inDim = flat.length / outDim;
            double[][] mat = new double[outDim][inDim];
            for (int o = 0; o < outDim; o++) {
                for (int i = 0; i < inDim; i++) {
                    mat[o][i] = flat[o * inDim + i];
                }
            }
            reshaped.put(weightName, mat);
        }
        weightMatrices = reshaped;
        weightsLoaded = true;
        Log.printLine("LIWSA-GNN: loaded " + weightMatrices.size() + " weight matrices and "
            + biasVectors.size() + " bias vectors from " + CONFIG_WEIGHTS_PATH);
    }

    /** Temporary holder: stashes the flat array as a 1-row matrix so it can
     * travel through the same map type until the second reshape pass runs. */
    private static double[][] reshapeLater(double[] flat) {
        return new double[][]{flat};
    }

    // -----------------------------------------------------------------
    // Forward pass -- direct, verified port of verify_plain_forward.py's
    // gnn_forward_plain(). See that file for the numeric cross-check
    // against the real trained PyTorch model.
    // -----------------------------------------------------------------
    private static double[] linear(double[] x, double[][] weight, double[] bias) {
        int outDim = weight.length;
        double[] out = new double[outDim];
        for (int o = 0; o < outDim; o++) {
            double s = bias[o];
            double[] row = weight[o];
            for (int i = 0; i < x.length; i++) {
                s += row[i] * x[i];
            }
            out[o] = s;
        }
        return out;
    }

    private static double[] relu(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.max(0.0, x[i]);
        }
        return out;
    }

    private static double[][] dagLayer(double[][] h, List<int[]> edges, int nNodes, int layerIdx) {
        int dim = h[0].length;
        int[] fanin = new int[nNodes];
        int[] fanout = new int[nNodes];
        for (int[] edge : edges) {
            fanin[edge[1]]++;
            fanout[edge[0]]++;
        }

        double[][] fromParents = new double[nNodes][dim];
        double[][] fromChildren = new double[nNodes][dim];
        for (int[] edge : edges) {
            int p = edge[0];
            int c = edge[1];
            double wPc = 1.0 / fanin[c];
            double wCp = 1.0 / fanout[p];
            for (int k = 0; k < dim; k++) {
                fromParents[c][k] += h[p][k] * wPc;
                fromChildren[p][k] += h[c][k] * wCp;
            }
        }

        String prefix = "layers." + layerIdx;
        double[][] selfW = weightMatrices.get(prefix + ".self_lin.weight");
        double[] selfB = biasVectors.get(prefix + ".self_lin.bias");
        double[][] parentW = weightMatrices.get(prefix + ".parent_lin.weight");
        double[] parentB = biasVectors.get(prefix + ".parent_lin.bias");
        double[][] childW = weightMatrices.get(prefix + ".child_lin.weight");
        double[] childB = biasVectors.get(prefix + ".child_lin.bias");

        double[][] newH = new double[nNodes][];
        for (int i = 0; i < nNodes; i++) {
            double[] a = linear(h[i], selfW, selfB);
            double[] b = linear(fromParents[i], parentW, parentB);
            double[] c = linear(fromChildren[i], childW, childB);
            double[] combined = new double[a.length];
            for (int k = 0; k < a.length; k++) {
                combined[k] = a[k] + b[k] + c[k];
            }
            newH[i] = relu(combined);
        }
        return newH;
    }

    /**
     * Runs the full GNN forward pass and returns {predictedMakespan,
     * predictedCost} (both normalised by sequential-execution time, the
     * same normalisation used during training -- see training_data.py's
     * docstring in the Python prototype for the rationale).
     */
    private static double[] gnnForward(double[][] nodeFeatures, List<int[]> edges) {
        int nNodes = nodeFeatures.length;
        double[][] h = nodeFeatures;
        for (int layer = 0; layer < N_LAYERS; layer++) {
            h = dagLayer(h, edges, nNodes, layer);
        }
        int dim = h[0].length;
        double[] graphEmbedding = new double[dim];
        for (int k = 0; k < dim; k++) {
            double sum = 0.0;
            for (int i = 0; i < nNodes; i++) {
                sum += h[i][k];
            }
            graphEmbedding[k] = sum / nNodes;
        }
        double[] x = linear(graphEmbedding, weightMatrices.get("head.0.weight"), biasVectors.get("head.0.bias"));
        x = relu(x);
        return linear(x, weightMatrices.get("head.3.weight"), biasVectors.get("head.3.bias"));
        // dropout is intentionally omitted: this is inference, and PyTorch's
        // own Dropout module is a no-op in eval() mode, matching this.
    }

    // -----------------------------------------------------------------
    // Feature construction (per-workflow-relative, matching
    // training_data.py's build_graph_features / sample_training_instance
    // exactly, so features seen at inference match what the model trained
    // on)
    // -----------------------------------------------------------------
    private double[][] buildNodeFeatures(int[] genotype) {
        int n = taskOrder.size();
        double maxLen = 0, maxDepthVal = 0, maxFanout = 1, maxFanin = 1;
        Map<Task, Integer> depthMap = new HashMap<>();
        for (Task t : taskOrder) {
            computeDepth(t, depthMap);
        }
        for (Task t : taskOrder) {
            maxLen = Math.max(maxLen, t.getCloudletTotalLength());
            maxDepthVal = Math.max(maxDepthVal, depthMap.get(t));
            maxFanout = Math.max(maxFanout, t.getChildList().size());
            maxFanin = Math.max(maxFanin, t.getParentList().size());
        }
        double maxMips = 0, maxCostRate = 0;
        for (CondorVM vm : vmList) {
            maxMips = Math.max(maxMips, vm.getMips());
            maxCostRate = Math.max(maxCostRate, vm.getCost());
        }

        double[][] features = new double[n][IN_DIM];
        for (int k = 0; k < n; k++) {
            Task t = taskOrder.get(k);
            CondorVM vm = vmList.get(genotype[k]);
            features[k][0] = t.getCloudletTotalLength() / maxLen;
            features[k][1] = depthMap.get(t) / Math.max(maxDepthVal, 1.0);
            features[k][2] = t.getChildList().size() / maxFanout;
            features[k][3] = t.getParentList().size() / maxFanin;
            features[k][4] = vm.getMips() / maxMips;
            features[k][5] = vm.getCost() / maxCostRate;
        }
        return features;
    }

    private List<int[]> buildEdgeList() {
        int n = taskOrder.size();
        Map<Task, Integer> idIndex = new HashMap<>();
        for (int k = 0; k < n; k++) {
            idIndex.put(taskOrder.get(k), k);
        }
        List<int[]> edges = new ArrayList<>();
        for (int k = 0; k < n; k++) {
            Task t = taskOrder.get(k);
            for (Object childObj : t.getChildList()) {
                Task child = (Task) childObj;
                Integer childIdx = idIndex.get(child);
                if (childIdx != null) {
                    edges.add(new int[]{k, childIdx});
                }
            }
        }
        return edges;
    }

    // -----------------------------------------------------------------
    // Seeding hook: sample candidate genotypes, score each with the GNN,
    // keep the best CONFIG_NUM_SEEDS_FROM_GNN of them.
    // -----------------------------------------------------------------
    @Override
    protected List<int[]> generateSeedGenotypes() {
        List<int[]> seeds = super.generateSeedGenotypes();

        loadWeightsIfNeeded();
        if (weightMatrices == null) {
            // load failed; already logged in loadWeightsIfNeeded()
            return seeds;
        }

        List<int[]> edges = buildEdgeList();
        int n = taskOrder.size();

        List<double[]> scored = new ArrayList<>();
        List<int[]> candidates = new ArrayList<>();

        for (int c = 0; c < CONFIG_NUM_CANDIDATES; c++) {
            int[] genotype = new int[n];
            for (int k = 0; k < n; k++) {
                genotype[k] = random.nextInt(vmList.size());
            }
            double[][] nodeFeatures = buildNodeFeatures(genotype);
            double[] pred = gnnForward(nodeFeatures, edges);
            candidates.add(genotype);
            scored.add(pred); // {predicted makespan, predicted cost}, both normalised
        }

        // Rank candidates by a swept makespan/cost weight, matching
        // LIWSA-ML's multi-trade-off seeding strategy (Eq. biasedSample in
        // the base paper), so the GNN seeds also span the trade-off axis
        // rather than all converging to the same greedy point.
        int numSeeds = Math.min(CONFIG_NUM_SEEDS_FROM_GNN, candidates.size());
        boolean[] taken = new boolean[candidates.size()];
        for (int s = 0; s < numSeeds; s++) {
            double wM = (numSeeds <= 1) ? 0.5 : 1.0 - (double) s / (numSeeds - 1);
            double wC = 1.0 - wM;
            int bestIdx = -1;
            double bestCombined = Double.POSITIVE_INFINITY;
            for (int c = 0; c < candidates.size(); c++) {
                if (taken[c]) {
                    continue;
                }
                double combined = wM * scored.get(c)[0] + wC * scored.get(c)[1];
                if (combined < bestCombined) {
                    bestCombined = combined;
                    bestIdx = c;
                }
            }
            if (bestIdx >= 0) {
                taken[bestIdx] = true;
                seeds.add(candidates.get(bestIdx));
            }
        }

        Log.printLine("LIWSA-GNN: scored " + CONFIG_NUM_CANDIDATES + " candidate genotypes, "
            + "seeded population with " + numSeeds + " GNN-selected genotypes spanning the "
            + "makespan/cost trade-off.");

        return seeds;
    }
}
