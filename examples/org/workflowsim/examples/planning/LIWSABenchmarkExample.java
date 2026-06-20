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
package org.workflowsim.examples.planning;

import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.HarddriveStorage;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.VmAllocationPolicySimple;
import org.cloudbus.cloudsim.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.workflowsim.CondorVM;
import org.workflowsim.Job;
import org.workflowsim.WorkflowDatacenter;
import org.workflowsim.WorkflowEngine;
import org.workflowsim.WorkflowPlanner;
import org.workflowsim.planning.LIWSAMLPlanningAlgorithm;
import org.workflowsim.planning.LIWSAPlanningAlgorithm;
import org.workflowsim.planning.MLEAOPlanningAlgorithm;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

/**
 * Benchmark driver: runs HEFT, Min-Min, MLEAO, LIWSA, and LIWSA-ML on a set
 * of DAX files and prints a comparison report.
 *
 * FIXES vs the previous version of this file:
 *  - Cost model: Parameters.setCostModel(Parameters.CostModel.VM) is now
 *    called explicitly. Without it, WorkflowSim defaults to CostModel.
 *    DATACENTER, which bills every job at one flat datacenter-wide rate
 *    instead of the rate of the VM it actually ran on, inflating reported
 *    costs and making every algorithm's cost look similar regardless of
 *    which VM types it actually chose.
 *  - Parameter injection: populationSize/generationCount/randomSeed are
 *    now set via public static CONFIG_* fields on the planning algorithm
 *    classes, read in their constructors. The previous reflection-based
 *    approach silently failed: WorkflowPlanner.processPlanning() builds
 *    and runs the planning algorithm as a local variable inside one
 *    method call, triggered only once CloudSim.startSimulation() begins
 *    processing events, so there was never a real object to configure.
 *
 * METRICS, beyond makespan and cost:
 *  - Pareto front size, hypervolume, and spread for the multi-objective
 *    algorithms (LIWSA, LIWSA-ML, MLEAO), captured via each algorithm's
 *    static `lastRun` snapshot.
 *  - Resource utilization and Jain's fairness index, computed uniformly
 *    for ALL five algorithms directly from the simulator's actual job
 *    results (not from internal algorithm state), so HEFT and Min-Min
 *    are included on equal footing.
 *  - Speedup relative to running the whole workflow sequentially on the
 *    single fastest VM type.
 *  - The three stochastic algorithms (MLEAO, LIWSA, LIWSA-ML) are each
 *    run across multiple random seeds and reported as mean +/- standard
 *    deviation, rather than a single run that could be an
 *    unrepresentative draw. HEFT and Min-Min are deterministic and are
 *    run once.
 */
public class LIWSABenchmarkExample {

    // ---------------------------------------------------------------
    // VM type table. { mips, bandwidthMbitS, costPerSec, ram_MB, storage_MB, count }
    // ---------------------------------------------------------------
    private static final double[][] VM_TYPES = {
        { 250.0, 160.0, 0.15, 512, 10000, 4},  // Micro  x4
        { 500.0, 160.0, 0.30, 512, 10000, 4},  // Small  x4
        {1000.0, 160.0, 0.60, 512, 10000, 4},  // Medium x4
        {2000.0, 160.0, 0.90, 512, 10000, 4},  // Large  x4
    };

    private static class RunResult {
        String name;
        double makespan;
        double cost;
        double hypervolume;
        int paretoFrontSize;
        double makespanSpread;
        double costSpread;
        double avgUtilization;
        double fairnessIndex;
        double speedup;
        long searchWallClockMillis;
        long totalSimWallClockMillis;
    }

    private static class Stats {
        double mean, std;

        static Stats of(List<Double> values) {
            Stats s = new Stats();
            int n = values.size();
            double sum = 0;
            for (double v : values) { sum += v; }
            s.mean = sum / n;
            double sq = 0;
            for (double v : values) { sq += (v - s.mean) * (v - s.mean); }
            s.std = n > 1 ? Math.sqrt(sq / (n - 1)) : 0.0;
            return s;
        }
    }

    public static void main(String[] args) {

        // ==============================================================
        // CONFIGURATION
        // ==============================================================

        // Standard-size workflows: fast to run, good for iterating.
        String[] standardDaxFiles = {
            "config/dax/Montage_50.xml",
            "config/dax/Montage_100.xml",
            "config/dax/CyberShake_30.xml",
            "config/dax/CyberShake_50.xml",
            "config/dax/Sipht_30.xml",
        };

        // Large workflows (~1000 tasks): include for the paper's final
        // scalability results. Each metaheuristic run will take
        // noticeably longer (more tasks -> more expensive decode calls
        // and, for LIWSA-ML, more predictor training rows). Comment any
        // of these out to skip if a quick iteration is needed. Confirm
        // exact filenames in your config/dax/ folder before running --
        // some WorkflowSim distributions name these slightly differently
        // (e.g. Epigenomics_997.xml vs Epigenomics_100.xml only).
        String[] largeDaxFiles = {
            "config/dax/Montage_1000.xml",
            "config/dax/CyberShake_1000.xml",
            "config/dax/Inspiral_1000.xml",
            "config/dax/Epigenomics_997.xml",
        };

        boolean includeLargeWorkflows = true;

        // Number of independent runs (different random seeds) for each
        // stochastic algorithm (MLEAO, LIWSA, LIWSA-ML). HEFT and Min-Min
        // are deterministic and are always run exactly once.
        int numSeeds = 5;
        long[] seeds = new long[numSeeds];
        for (int i = 0; i < numSeeds; i++) { seeds[i] = i + 1; }

        int populationSize = 30;
        int generationCount = 100;

        // ==============================================================
        // END CONFIGURATION
        // ==============================================================

        List<String> daxFiles = new ArrayList<>();
        for (String f : standardDaxFiles) { daxFiles.add(f); }
        if (includeLargeWorkflows) {
            for (String f : largeDaxFiles) { daxFiles.add(f); }
        }

        DecimalFormat df2 = new DecimalFormat("#####0.00");
        DecimalFormat df1 = new DecimalFormat("#####0.0");
        DecimalFormat df3 = new DecimalFormat("0.000");

        for (String daxPath : daxFiles) {
            if (!new File(daxPath).exists()) {
                System.out.println("Skipping (not found): " + daxPath);
                continue;
            }
            String workflowName = new File(daxPath).getName().replace(".xml", "");
            System.out.println();
            System.out.println("=".repeat(78));
            System.out.println("WORKFLOW: " + workflowName);
            System.out.println("=".repeat(78));

            // Deterministic baselines: one run each
            RunResult heft = runPlanning(daxPath, Parameters.PlanningAlgorithm.HEFT,
                Parameters.SchedulingAlgorithm.STATIC, "HEFT", 0L,
                populationSize, generationCount);

            RunResult minmin = runPlanning(daxPath, Parameters.PlanningAlgorithm.INVALID,
                Parameters.SchedulingAlgorithm.MINMIN, "Min-Min", 0L,
                populationSize, generationCount);

            // Stochastic algorithms: one run per seed
            List<RunResult> mleaoRuns = new ArrayList<>();
            List<RunResult> liwsaRuns = new ArrayList<>();
            List<RunResult> liwsaMlRuns = new ArrayList<>();

            for (long seed : seeds) {
                RunResult m = runPlanning(daxPath, Parameters.PlanningAlgorithm.MLEAO,
                    Parameters.SchedulingAlgorithm.STATIC, "MLEAO", seed,
                    populationSize, generationCount);
                if (m != null) { mleaoRuns.add(m); }

                RunResult l = runPlanning(daxPath, Parameters.PlanningAlgorithm.LIWSA,
                    Parameters.SchedulingAlgorithm.STATIC, "LIWSA", seed,
                    populationSize, generationCount);
                if (l != null) { liwsaRuns.add(l); }

                RunResult lm = runPlanning(daxPath, Parameters.PlanningAlgorithm.LIWSAML,
                    Parameters.SchedulingAlgorithm.STATIC, "LIWSA-ML", seed,
                    populationSize, generationCount);
                if (lm != null) { liwsaMlRuns.add(lm); }
            }

            // ---- table ----
            System.out.println();
            System.out.printf("%-10s %14s %14s %7s %10s %7s %7s %8s%n",
                "Algorithm", "Makespan(s)", "Cost", "Front", "Hypervol.",
                "Util%", "Fair", "Speedup");
            System.out.println("-".repeat(86));

            printSingle(df2, df1, df3, heft);
            printSingle(df2, df1, df3, minmin);
            printAggregate(df2, df1, df3, "MLEAO", mleaoRuns);
            printAggregate(df2, df1, df3, "LIWSA", liwsaRuns);
            printAggregate(df2, df1, df3, "LIWSA-ML", liwsaMlRuns);

            // ---- improvement ratios (means, for the stochastic algorithms) ----
            if (heft != null && !liwsaRuns.isEmpty() && !mleaoRuns.isEmpty() && !liwsaMlRuns.isEmpty()) {
                double liwsaMk = mean(liwsaRuns, r -> r.makespan);
                double liwsaCost = mean(liwsaRuns, r -> r.cost);
                double mleaoMk = mean(mleaoRuns, r -> r.makespan);
                double mleaoCost = mean(mleaoRuns, r -> r.cost);
                double mlMk = mean(liwsaMlRuns, r -> r.makespan);
                double mlCost = mean(liwsaMlRuns, r -> r.cost);

                System.out.println();
                System.out.printf("  LIWSA    vs HEFT:  makespan %+6.1f%%  cost %+6.1f%%%n",
                    pct(liwsaMk, heft.makespan), pct(liwsaCost, heft.cost));
                System.out.printf("  LIWSA-ML vs HEFT:  makespan %+6.1f%%  cost %+6.1f%%%n",
                    pct(mlMk, heft.makespan), pct(mlCost, heft.cost));
                System.out.printf("  LIWSA-ML vs MLEAO: makespan %+6.1f%%  cost %+6.1f%%%n",
                    pct(mlMk, mleaoMk), pct(mlCost, mleaoCost));
                System.out.printf("  LIWSA-ML vs LIWSA: makespan %+6.1f%%  cost %+6.1f%%%n",
                    pct(mlMk, liwsaMk), pct(mlCost, liwsaCost));
            }
        }
    }

    private static double pct(double value, double base) {
        return 100.0 * (value - base) / base;
    }

    private interface Extractor { double get(RunResult r); }

    private static double mean(List<RunResult> runs, Extractor ex) {
        double sum = 0;
        for (RunResult r : runs) { sum += ex.get(r); }
        return sum / runs.size();
    }

    private static void printSingle(DecimalFormat df2, DecimalFormat df1, DecimalFormat df3, RunResult r) {
        if (r == null) { return; }
        System.out.printf("%-10s %14s %14s %7d %10s %7s %7s %8s%n",
            r.name, df2.format(r.makespan), df2.format(r.cost), r.paretoFrontSize,
            df1.format(r.hypervolume), df1.format(r.avgUtilization * 100),
            df3.format(r.fairnessIndex), df3.format(r.speedup));
    }

    private static void printAggregate(DecimalFormat df2, DecimalFormat df1, DecimalFormat df3,
                                        String label, List<RunResult> runs) {
        if (runs.isEmpty()) {
            System.out.printf("%-10s  (no successful runs)%n", label);
            return;
        }
        List<Double> mk = new ArrayList<>(), cost = new ArrayList<>(), hv = new ArrayList<>();
        List<Double> util = new ArrayList<>(), fair = new ArrayList<>(), speed = new ArrayList<>();
        List<Double> frontSize = new ArrayList<>();
        for (RunResult r : runs) {
            mk.add(r.makespan); cost.add(r.cost); hv.add(r.hypervolume);
            util.add(r.avgUtilization); fair.add(r.fairnessIndex); speed.add(r.speedup);
            frontSize.add((double) r.paretoFrontSize);
        }
        Stats mkS = Stats.of(mk), costS = Stats.of(cost), hvS = Stats.of(hv);
        Stats utilS = Stats.of(util), fairS = Stats.of(fair), speedS = Stats.of(speed);
        Stats frontS = Stats.of(frontSize);

        String mkStr = df2.format(mkS.mean) + "+-" + df1.format(mkS.std);
        String costStr = df2.format(costS.mean) + "+-" + df1.format(costS.std);

        System.out.printf("%-10s %14s %14s %7s %10s %7s %7s %8s%n",
            label, mkStr, costStr,
            df1.format(frontS.mean),
            df1.format(hvS.mean),
            df1.format(utilS.mean * 100),
            df3.format(fairS.mean),
            df3.format(speedS.mean));
    }

    /**
     * Runs one complete WorkflowSim simulation and returns the result,
     * including the universal metrics (utilization, fairness, speedup)
     * computed from the actual job results, and, for the multi-objective
     * algorithms, the Pareto front metrics captured via their static
     * lastRun snapshot.
     */
    private static RunResult runPlanning(
            String daxPath,
            Parameters.PlanningAlgorithm planningAlg,
            Parameters.SchedulingAlgorithm schedulingAlg,
            String label, long seed,
            int populationSize, int generationCount) {

        try {
            // Static configuration, read by the algorithm's no-arg
            // constructor at the moment WorkflowPlanner instantiates it
            // internally. This is the only injection point that actually
            // reaches a WorkflowSim-driven run -- see the class-level
            // comment for why.
            LIWSAPlanningAlgorithm.CONFIG_POPULATION_SIZE = populationSize;
            LIWSAPlanningAlgorithm.CONFIG_GENERATION_COUNT = generationCount;
            LIWSAPlanningAlgorithm.CONFIG_RANDOM_SEED = seed;
            MLEAOPlanningAlgorithm.CONFIG_POPULATION_SIZE = populationSize;
            MLEAOPlanningAlgorithm.CONFIG_GENERATION_COUNT = generationCount;
            MLEAOPlanningAlgorithm.CONFIG_RANDOM_SEED = seed;

            int totalVMs = 0;
            for (double[] t : VM_TYPES) { totalVMs += (int) t[5]; }

            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
            ClusteringParameters cp = new ClusteringParameters(
                0, 0, ClusteringParameters.ClusteringMethod.NONE, null);

            Parameters.init(totalVMs, daxPath, null, null, op, cp,
                schedulingAlg, planningAlg, null, 0);

            // THE COST FIX: without this, WorkflowSim bills every job at
            // the flat datacenter rate instead of its VM's actual rate.
            Parameters.setCostModel(Parameters.CostModel.VM);

            ReplicaCatalog.init(ReplicaCatalog.FileSystem.LOCAL);
            CloudSim.init(1, Calendar.getInstance(), false);

            WorkflowDatacenter dc = createDatacenter("DC_" + label + "_" + seed);
            WorkflowPlanner wfPlanner = new WorkflowPlanner("planner_" + label + "_" + seed, 1);
            WorkflowEngine wfEngine = wfPlanner.getWorkflowEngine();
            List<CondorVM> vmList = createVMs(wfEngine.getSchedulerId(0));
            wfEngine.submitVmList(vmList, 0);
            wfEngine.bindSchedulerDatacenter(dc.getId(), 0);

            long simStart = System.currentTimeMillis();
            CloudSim.startSimulation();
            List<Job> jobs = wfEngine.getJobsReceivedList();
            CloudSim.stopSimulation();
            long simWall = System.currentTimeMillis() - simStart;

            RunResult r = new RunResult();
            r.name = label;
            r.totalSimWallClockMillis = simWall;

            // ---- universal metrics: makespan, cost, utilization, fairness,
            //      speedup -- computed identically for all 5 algorithms,
            //      directly from the simulator's actual results. ----
            double[] vmBusyTime = new double[vmList.size()];
            double totalTaskLength = 0.0;
            for (Job job : jobs) {
                if (job.getClassType() == org.workflowsim.utils.Parameters.ClassType.STAGE_IN.value) {
                    continue;
                }
                r.makespan = Math.max(r.makespan, job.getFinishTime());
                r.cost += job.getActualCPUTime() * job.getCostPerSec();
                int vmId = job.getVmId();
                if (vmId >= 0 && vmId < vmBusyTime.length) {
                    vmBusyTime[vmId] += job.getActualCPUTime();
                }
                totalTaskLength += job.getCloudletTotalLength();
            }

            double busySum = 0, busySqSum = 0;
            for (double b : vmBusyTime) { busySum += b; busySqSum += b * b; }
            r.avgUtilization = (r.makespan > 0)
                ? (busySum / vmList.size()) / r.makespan : 0.0;
            // Jain's fairness index over per-VM busy time: 1.0 = perfectly
            // even load across all VMs, 1/numVMs = maximally unfair
            // (all load concentrated on one VM).
            r.fairnessIndex = (busySqSum > 0)
                ? (busySum * busySum) / (vmList.size() * busySqSum) : 0.0;

            double fastestMips = 0;
            for (double[] t : VM_TYPES) { fastestMips = Math.max(fastestMips, t[0]); }
            double sequentialTime = totalTaskLength / fastestMips;
            r.speedup = (r.makespan > 0) ? sequentialTime / r.makespan : 0.0;

            // ---- multi-objective metrics: only meaningful for the
            //      Pareto-front algorithms. HEFT/Min-Min report a
            //      single-point "front" of size 1 for comparability. ----
            List<double[]> frontPoints = null;
            long searchWall = 0;
            if ((planningAlg == Parameters.PlanningAlgorithm.LIWSA
                    || planningAlg == Parameters.PlanningAlgorithm.LIWSAML)
                    && LIWSAPlanningAlgorithm.lastRun != null) {
                frontPoints = LIWSAPlanningAlgorithm.lastRun.paretoFrontPoints;
                searchWall = LIWSAPlanningAlgorithm.lastRun.searchWallClockMillis;
            } else if (planningAlg == Parameters.PlanningAlgorithm.MLEAO
                    && MLEAOPlanningAlgorithm.lastRun != null) {
                frontPoints = MLEAOPlanningAlgorithm.lastRun.paretoFrontPoints;
                searchWall = MLEAOPlanningAlgorithm.lastRun.searchWallClockMillis;
            }
            r.searchWallClockMillis = searchWall;

            if (frontPoints == null) {
                frontPoints = new ArrayList<>();
                frontPoints.add(new double[]{r.makespan, r.cost});
            }
            r.paretoFrontSize = frontPoints.size();

            // Reference point for hypervolume: 20% beyond the worst point
            // on this algorithm's own front. For comparing hypervolume
            // ACROSS algorithms with full rigor, recompute externally
            // using one shared reference point spanning every algorithm's
            // results for the same workflow; this per-run value is still
            // a valid standalone richness indicator for one algorithm.
            double maxM = 0, maxC = 0, minM = Double.MAX_VALUE, minC = Double.MAX_VALUE;
            for (double[] p : frontPoints) {
                maxM = Math.max(maxM, p[0]); maxC = Math.max(maxC, p[1]);
                minM = Math.min(minM, p[0]); minC = Math.min(minC, p[1]);
            }
            double refM = maxM * 1.2 + 1e-6;
            double refC = maxC * 1.2 + 1e-6;
            r.hypervolume = hypervolume2D(frontPoints, refM, refC);
            r.makespanSpread = maxM - minM;
            r.costSpread = maxC - minC;

            return r;

        } catch (Exception e) {
            System.err.println("[" + label + "] simulation failed: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * 2D hypervolume for minimization of both objectives, against a fixed
     * reference point assumed to be dominated by (worse than) every point
     * in the front. Direct port of the Python prototype's hypervolume_2d,
     * validated there against a hand-computed staircase example.
     */
    private static double hypervolume2D(List<double[]> points, double refM, double refC) {
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

    private static List<CondorVM> createVMs(int userId) {
        LinkedList<CondorVM> list = new LinkedList<>();
        int vmId = 0;
        for (double[] type : VM_TYPES) {
            double mips = type[0]; long bw = (long) type[1];
            double cost = type[2]; int ram = (int) type[3];
            long storage = (long) type[4]; int count = (int) type[5];
            for (int k = 0; k < count; k++) {
                list.add(new CondorVM(vmId++, userId, mips, 1, ram, bw, storage,
                    "Xen", cost, 0.0, 0.0, 0.0, new CloudletSchedulerSpaceShared()));
            }
        }
        return list;
    }

    private static WorkflowDatacenter createDatacenter(String name) {
        int totalVMs = 0;
        for (double[] t : VM_TYPES) { totalVMs += (int) t[5]; }
        int numHosts = Math.max(1, (totalVMs + 3) / 4);

        List<Host> hostList = new ArrayList<>();
        for (int h = 0; h < numHosts; h++) {
            List<Pe> peList = new ArrayList<>();
            for (int p = 0; p < 16; p++) {
                peList.add(new Pe(p, new PeProvisionerSimple(2000)));
            }
            hostList.add(new Host(h,
                new RamProvisionerSimple(8192),
                new BwProvisionerSimple(100000),
                1000000L, peList,
                new VmSchedulerTimeShared(peList)));
        }

        DatacenterCharacteristics dc = new DatacenterCharacteristics(
            "x86", "Linux", "Xen", hostList, 10.0, 3.0, 0.05, 0.1, 0.1);
        WorkflowDatacenter datacenter = null;
        try {
            HarddriveStorage s = new HarddriveStorage(name, 1e12);
            s.setMaxTransferRate(100);
            LinkedList<Storage> sl = new LinkedList<>();
            sl.add(s);
            datacenter = new WorkflowDatacenter(name, dc,
                new VmAllocationPolicySimple(hostList), sl, 0);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return datacenter;
    }
}
