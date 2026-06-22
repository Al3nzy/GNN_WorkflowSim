/**
 * Copyright 2025-2026 SDU University, Kazakhstan
 * @author Dr. Mohammed Alaa Ala'anzy
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
package org.workflowsim.examples.planning;

import java.util.ArrayList;
import java.util.List;

/**
 * 2D hypervolume for minimization of both objectives (makespan, cost).
 *
 * IMPORTANT, learned the hard way from a real benchmark run: hypervolume
 * is only meaningful when every front being compared is measured against
 * the SAME reference point. Computing each algorithm's reference point
 * from its own worst value (e.g. "1.2x this front's own max") makes a
 * single bad point look artificially "good": a single-point front with a
 * huge makespan and cost will mechanically produce a huge hypervolume
 * number under that scheme, purely as an artifact of scale, not because
 * it represents a richer or better set of solutions. Always compute ONE
 * shared reference point across every algorithm/seed being compared for
 * the same workflow (sharedReferencePoint below), and use that same
 * point for all of them.
 */
public class ParetoMetrics {

    /**
     * Computes a single reference point, 20% beyond the worst makespan
     * and worst cost seen across every point in every front passed in.
     * Call this once per workflow, after collecting every algorithm's
     * (and every seed's) front, then reuse the result for every
     * hypervolume calculation for that workflow.
     */
    public static double[] sharedReferencePoint(List<List<double[]>> allFronts) {
        double maxM = 0, maxC = 0;
        for (List<double[]> front : allFronts) {
            for (double[] p : front) {
                maxM = Math.max(maxM, p[0]);
                maxC = Math.max(maxC, p[1]);
            }
        }
        return new double[]{maxM * 1.2 + 1e-6, maxC * 1.2 + 1e-6};
    }

    /**
     * 2D hypervolume of `points` against `ref`, assuming ref is
     * dominated by (worse than) every point. Validated against a
     * hand-computed staircase example during the Python prototype phase.
     */
    public static double hypervolume2D(List<double[]> points, double refM, double refC) {
        List<double[]> sorted = new ArrayList<>(points);
        sorted.sort((a, b) -> Double.compare(a[0], b[0]));
        double hv = 0.0;
        for (int i = 0; i < sorted.size(); i++) {
            double m = sorted.get(i)[0];
            double c = sorted.get(i)[1];
            if (m >= refM || c >= refC) { continue; }
            double nextM = (i + 1 < sorted.size()) ? sorted.get(i + 1)[0] : refM;
            if (nextM > m) {
                hv += (nextM - m) * (refC - c);
            }
        }
        return hv;
    }
}
